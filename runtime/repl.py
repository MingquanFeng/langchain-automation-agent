
"""
交互式终端（REPL）：在命令行里像聊天一样给 Agent 下指令。

课件未覆盖，属于 runtime 层的外壳能力 —— 和 scheduler.py、server.py 是同一层：
它们都只负责"什么时候、怎么触发"，Agent 本体仍由 agent/graph.py 的 build_agent 装配。

对应课件：
课件未覆盖交互式终端，需要自行实现。

主要职责：
1. 在终端里持续接收指令并流式展示 Agent 的回答。
2. 提供斜杠命令控制会话、写权限、思考过程显示。
3. 过滤掉噪音（模型思考过程、中间件例行提示），让终端可读。

用法：
    python -m runtime chat

设计要点：
- 复用 runtime.runner 里的 agent 单例，checkpointer 照常落盘，记忆跨进程保留。
- 斜杠命令做控制（换会话、开关写权限等），其余输入都当作给 Agent 的指令。
- 默认只读：写权限跟 /run 一样要显式开（/write on），避免误改 data/ 下的文件。

注意：
和 serve 子命令的区别：
    serve：起 HTTP 服务，无人值守，靠接口触发。
    chat：起交互式终端，人在回路里逐条下指令。
两者共用 runner 里的 Agent 单例和任务上下文规则，
但 chat 不起调度器，也不监听任何端口。
"""

from __future__ import annotations

# re：标准库的正则模块。
import re

# sys：标准库，用来临时替换 stdout。
import sys

# uuid：标准库，用来生成随机会话 ID。
import uuid

# AIMessage / AIMessageChunk：LangChain 的模型消息类型。
# 前者是完整消息，后者是流式增量消息。
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

# RUNTIME：应用运行配置，这里取通知渠道。
from agent.config import RUNTIME

# AgentContext：单次调用的运行时上下文结构。
from agent.context import AgentContext

# get_agent：进程内唯一的 Agent 单例。
from .runner import get_agent

# MiniMax 这类推理模型会把思考过程以 <think>...</think> 的形式混在 content 里返回，
# 默认滤掉，否则终端会被大段自说自话刷屏。需要看推理过程时用 /think on。
#
# 正则说明：
# .*? 是非贪婪匹配，尽量少匹配。
# (?:</think>|\Z) 表示匹配到闭标签，或者一直匹配到字符串结尾（闭标签还没到的情况）。
# re.S 让 . 能匹配换行符，否则跨行的思考块会被漏掉。
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)

# 这些中间件注入的消息是每轮的例行提示，终端上显示写权限就够了，不重复刷屏
#
# 取值是 langgraph_node 名字的前缀，
# 用 startswith 比较是因为同一个节点可能带后缀。
_HINT_NODES = ("ContextInjectionMiddleware",)

# HELP：斜杠命令的帮助文本。
# 以 \ 结尾的续行符让这个字符串紧贴左花括号，
# 避免第一行前面多出一个空格。
HELP = """\
可用命令：
  /help            显示这份帮助
  /new             开一个新会话（换 thread_id，上下文清空）
  /thread          显示当前会话 ID
  /write on|off    开关写权限（默认 off，只读）
  /think on|off    是否显示模型思考过程（默认 off）
  /exit            退出（Ctrl-D、Ctrl-C 同效）

其余输入都会作为指令发给 Agent。
记得带上 thread_id 的会话会保留上下文，可以连续追问。
"""


# ============================================================
# 1. 思考过程过滤器
# ============================================================

class _ThinkFilter:
    """流式滤掉 <think> 块。

    逐 chunk 拼接后再判断；闭标签还没到时也先把当前这段当思考内容滤掉，
    避免思考过程漏到终端上。

    为什么不能逐 chunk 单独判断：
    思考内容是分片送过来的，
    某个 chunk 可能正好停在 "<think>" 中间，
    单独判断会以为没有思考块而把它原样输出。
    所以要把历史片段都留着，每次拿完整缓冲重新过滤。

    类名以 _ 开头：
        表示模块内部使用，
        这是 Python 命名约定，不是访问限制。
    """

    def __init__(self) -> None:
        # 累积收到的原始文本（含思考块）。
        self._buf = ""

        # 已经输出过的干净文本长度。
        # 用它做切片，避免重复输出、也避免漏输出。
        self._printed = 0

    def reset(self) -> None:
        """
        清空缓冲。

        每轮对话开始前调用，
        防止上一轮的思考片段混进这一轮。
        """
        self._buf = ""
        self._printed = 0

    def feed(self, text: str) -> str:
        """吃进一段增量文本，返回可以显示的部分。

        参数：
            text：新收到的增量文本。

        返回：
            本次新产生的、可以显示的干净文本。
            没有新内容时返回空字符串。

        注意：
        返回空字符串不代表出错，
        很可能只是这一段全是思考内容。
        """
        self._buf += text

        # 拿完整缓冲重新过滤。
        clean = _THINK_RE.sub("", self._buf)

        # 长度没变说明新来的部分全在思考块里，没有可显示内容。
        if len(clean) <= self._printed:
            return ""

        # 只吐出新增的那一段。
        out = clean[self._printed :]
        self._printed = len(clean)
        return out


# ============================================================
# 2. 输出代理
# ============================================================

class _LineGuard:
    """输出代理：让第三方 print 独占一行。

    中间件的统计、notify 的输出都是直接 print 到 stdout 的，流式回答的过程中
    混进来会粘在回答末尾。这个代理保证它们另起一行，同时不影响我们自己写的
    流式内容（走 raw）。

    为什么要代理 stdout 而不是改 notify 的实现：
    notify 和中间件是 agent 层的能力，
    它们不该知道「当前是不是在终端里」。
    在 runtime 层临时把 sys.stdout 换掉，
    既不用改 agent 层，也不影响其他入口。

    类名以 _ 开头：
        表示模块内部使用，
        这是 Python 命名约定，不是访问限制。
    """

    def __init__(self, stream) -> None:
        # 被代理的原始流。
        self._stream = stream

        # 当前是否在行首。
        # 为 True 表示可以直接写内容而不需要先换行。
        self.at_line_start = True

    def write(self, text: str) -> int:
        """
        代理 write()：写出去的内容独占一行。

        参数：
            text：要写出的文本。

        返回：
            写出的字符数（符合文件对象的 write 约定）。

        注意：
        和 raw() 的区别：
        write() 会先保证另起一行，
        raw() 原样输出。
        """
        # 已经在行中间了，又不是以换行开头，先补一个换行。
        if text and not self.at_line_start and not text.startswith("\n"):
            self._stream.write("\n")
            self.at_line_start = True
        self._stream.write(text)
        if text:
            # 只有文本以换行结尾，当前才回到行首。
            self.at_line_start = text.endswith("\n")
        return len(text)

    def raw(self, text: str) -> None:
        """自己写的内容：原样输出，不强制换行，只更新行首状态。

        用于流式回答的增量文本，
        换行会打断阅读节奏。
        """
        self._stream.write(text)
        if text:
            self.at_line_start = text.endswith("\n")

    def line(self, text: str) -> None:
        """写一整行：必要时先换行。

        用于工具调用提示这类需要独立成行的内容。
        """
        if not self.at_line_start:
            self._stream.write("\n")
            self.at_line_start = True
        self._stream.write(text)
        if text:
            self.at_line_start = text.endswith("\n")

    def flush(self) -> None:
        """转发 flush() 到原始流。"""
        self._stream.flush()

    def __getattr__(self, name: str):
        """
        把没定义的方法转发给原始流。

        __getattr__ 只在常规属性查找失败时触发，
        所以 write / raw / line 等自己定义的方法不受影响，
        而 isatty()、encoding 这些没定义的方法会落到这里，
        转发给真实的 stdout。

        注意：
        用 getattr 而不是直接赋值，
        是为了保持「代理」而不是「复制」——
        原始流的行为变化能透过来。
        """
        return getattr(self._stream, name)


# ============================================================
# 3. 终端会话
# ============================================================

class ReplSession:
    """一次终端会话。

    持有这次会话的全部状态：
    当前 thread_id、写权限、是否显示思考过程、思考过滤器。
    """

    def __init__(self) -> None:
        # 复用 runner 里的 Agent 单例，
        # 这样 checkpointer 照常落盘，记忆跨进程保留。
        self.agent = get_agent()

        # 固定 thread_id，重启终端还能接着上次聊；/new 换新的
        #
        # 用固定的 "cli-chat" 而不是随机 ID，
        # 正是因为要跨进程续聊。
        # 随机 ID 的话每次启动都是全新会话，
        # 上次关掉终端前聊到哪就全丢了。
        self.thread_id = "cli-chat"

        # 写权限默认关闭，只读。
        # 和 /run、/run/stream 的默认值一致。
        self.allow_write = False

        # 默认不显示模型的思考过程。
        self.show_thinking = False

        # 思考过程过滤器，每轮对话前 reset。
        self._think = _ThinkFilter()

    # ---------- 交互主循环 ----------

    def loop(self) -> None:
        """
        交互主循环。

        流程：
        1. 打印欢迎信息。
        2. 循环读输入。
        3. 以 "/" 开头当命令处理，其余当指令发给 Agent。
        4. Ctrl-D（EOFError）或 Ctrl-C（KeyboardInterrupt）退出。

        注意：
        这里用 input() 而不是异步读 stdin，
        因为整个终端是单线程顺序执行的，
        流式输出也是同步迭代，异步反而更绕。
        """
        print("=" * 64)
        print("自动化 Agent 终端。斜杠命令输入 /help 查看。")
        print(f"会话 ID：{self.thread_id}    写权限：{'允许' if self.allow_write else '禁止（只读）'}")
        print("=" * 64)
        while True:
            try:
                line = input("› ").strip()
            except (EOFError, KeyboardInterrupt):
                # Ctrl-D 抛 EOFError，Ctrl-C 抛 KeyboardInterrupt。
                # 两者都当作「用户要走了」处理。
                print()
                break
            if not line:
                # 空行直接跳过，不当成指令。
                continue
            if line.startswith("/"):
                # command() 返回 False 表示要退出。
                if not self.command(line):
                    break
                continue
            self.send(line)
        print("再见。")

    def command(self, line: str) -> bool:
        """执行斜杠命令。返回 False 表示要退出。

        参数：
            line：整行输入，以 "/" 开头。

        返回：
            True 表示继续留在终端；
            False 表示退出主循环。

        注意：
        命令解析是手写的，不用 argparse 之类的框架，
        因为斜杠命令很简单，参数固定是 on/off 两个词。
        """
        # 去掉开头的 "/"，再按空白切分。
        parts = line[1:].split()

        # 命令名转小写，允许 /EXIT、/Exit。
        name = parts[0].lower() if parts else ""

        # 第二个词是参数，转小写，允许 /write ON。
        arg = parts[1].lower() if len(parts) > 1 else ""

        if name in ("exit", "quit", "q"):
            return False
        if name in ("help", "h", "?"):
            print(HELP)
        elif name == "new":
            # 换一个新会话。
            # 换 thread_id 就等于换了一条 Checkpointer 里的历史，
            # 旧会话的上下文还在数据库里，只是当前不再引用它。
            self.thread_id = f"chat-{uuid.uuid4().hex[:8]}"
            print(f"已切换到新会话：{self.thread_id}")
        elif name == "thread":
            # 显示当前会话 ID，方便外部知道该用哪个 thread_id 接续。
            print(self.thread_id)
        elif name == "write":
            # 开关写权限。
            # 只认 on/off/true/false/1/0 这些明确的值，
            # 其他一律提示用法，避免手滑开错。
            if arg in ("on", "true", "1"):
                self.allow_write = True
            elif arg in ("off", "false", "0"):
                self.allow_write = False
            else:
                print("用法：/write on|off")
                return True
            print(f"写权限：{'允许' if self.allow_write else '禁止'}")
        elif name == "think":
            # 开关思考过程显示。
            if arg in ("on", "true", "1"):
                self.show_thinking = True
            elif arg in ("off", "false", "0"):
                self.show_thinking = False
            else:
                print("用法：/think on|off")
                return True
            print(f"显示思考过程：{'开' if self.show_thinking else '关'}")
        else:
            print(f"未知命令：/{name}，输入 /help 查看")
        return True

    # ---------- 发一条指令给 Agent ----------

    def send(self, instruction: str) -> None:
        """
        把一条指令发给 Agent，并流式展示回答。

        参数：
            instruction：用户的自然语言指令。

        注意：
        这里没有走 runner.run_task()，
        因为流式需要自己控制 agent.stream() 的迭代。
        但上下文字段和 thread_id 的构造规则和 run_task() 保持一致。
        """
        ctx = AgentContext(
            task_name="cli-chat",
            trigger="cli",
            allow_write=self.allow_write,
            notify_channel=RUNTIME.notify_channel,
        )

        # thread_id 放在 config["configurable"] 里，
        # 这是 Checkpointer 识别会话的依据。
        config = {"configurable": {"thread_id": self.thread_id}}

        # 每轮重置思考过滤器，
        # 防止上一轮的思考片段影响这一轮的过滤位置计算。
        self._think.reset()

        # 记录已经显示过的工具名，避免同一次任务里重复提示。
        seen_tools: set[str] = set()

        # 是否已经打印过回答起始标记 🤖。
        started = False

        # 中间件和 notify 都是直接 print 的，装个代理让它们别粘在回答后面
        original_stdout = sys.stdout
        guard = _LineGuard(original_stdout)

        # 临时把 sys.stdout 换成代理。
        # 这样 agent 层里所有 print 都自动走代理。
        sys.stdout = guard
        try:
            # 两个流一起要：messages 拿增量文本，values 里能看到完整的消息列表，
            # 工具调用只能从 values 里认（messages 模式下非流式模型不吐 tool_call_chunks）
            # 多流模式的每一项是 (模式名, 载荷)
            #
            # 具体来说：
            # stream_mode 是列表时，agent.stream() 迭代出的每一项
            # 都是一个二元组 (mode, payload)，
            # mode 是字符串（"messages" / "values"），
            # payload 的结构由 mode 决定。
            #
            # "messages" 模式：
            # payload 是 (chunk, metadata) 二元组，
            # chunk 是 AIMessageChunk（增量）或 AIMessage（整条），
            # metadata 里带 langgraph_node 等信息。
            #
            # "values" 模式：
            # payload 是整个状态字典，里面有完整的 messages 列表。
            for mode, payload in self.agent.stream(
                {"messages": [HumanMessage(content=instruction)]},
                config,

                # 注意：
                # context 是 stream 的独立参数，不在 config 里。
                # 写进 config 里的话 runtime.context 会是 None。
                context=ctx,
                stream_mode=["messages", "values"],
            ):
                # --------------------------------------------------------
                # 4.1 values 流：识别工具调用
                # --------------------------------------------------------

                if mode == "values":
                    # values 模式给的是完整状态，
                    # 工具调用信息只在 messages 列表里的 AIMessage.tool_calls 上。
                    for m in payload.get("messages", []):
                        if isinstance(m, AIMessage) and m.tool_calls:
                            for tc in m.tool_calls:
                                name = tc.get("name")
                                # 同一个工具名只提示一次。
                                if name and name not in seen_tools:
                                    seen_tools.add(name)
                                    guard.line(f"  ⚙️  {name}\n")
                    # values 分支处理完就跳过下面的 messages 处理。
                    continue

                # --------------------------------------------------------
                # 4.2 messages 流：展示模型输出
                # --------------------------------------------------------

                # 只关心 messages 模式。
                # stream_mode 是列表时可能还有别的模式，统一跳过。
                if mode != "messages":
                    continue

                # messages 模式的 payload 是 (chunk, metadata)。
                chunk, meta = payload

                if isinstance(chunk, AIMessageChunk):
                    # 增量文本。
                    # content 可能是字符串，也可能是内容块列表，
                    # 这里只处理字符串形态。
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if not text:
                        continue
                    # show_thinking 为 True 时原样输出，
                    # 否则过一遍过滤器把 <think> 块去掉。
                    out = text if self.show_thinking else self._think.feed(text)
                elif isinstance(chunk, AIMessage):
                    # 整条消息（模型最终回答、中间件的错误提示等）。
                    # ContextInjectionMiddleware 每轮都插一句只读提示，终端自己
                    # 已经显示写权限了，这句是噪音，滤掉；其余（写操作被拦截的
                    # 说明、模型调用失败等）要显示。
                    node = str(meta.get("langgraph_node", ""))
                    if node.startswith(_HINT_NODES):
                        continue
                    out = chunk.content if isinstance(chunk.content, str) else ""
                else:
                    # HumanMessage 之类的非模型消息，忽略。
                    continue

                if not out:
                    continue

                # 第一次有内容时补上 🤖 标记。
                if not started:
                    out = out.lstrip("\n")  # 模型回答常带前导换行，别顶掉 🤖
                    if not out:
                        continue
                    guard.raw("🤖 ")
                    started = True
                guard.raw(out)
        except Exception as exc:  # 别让一次失败把整个终端带崩
            # 终端是交互式的，一次失败不该把会话整个结束。
            # 打印错误后继续等下一条输入。
            guard.line(f"调用失败：{type(exc).__name__}: {exc}\n")
            return
        finally:
            # 无论正常结束还是抛异常，都要还原 stdout。
            # 少了这一步，后面所有 print 都会打到代理上。
            sys.stdout = original_stdout

        # 回答结束后补一个换行，
        # 让下一条输入提示符另起一行。
        if started:
            print()


# ============================================================
# 4. 终端入口
# ============================================================

def run_repl() -> None:
    """终端入口。由 runtime/__main__.py 的 chat 子命令调用。

    注意：
    这里只创建会话并进入主循环，
    不做任何配置读取和资源清理，
    因为进程退出时这些都会被释放。
    """
    ReplSession().loop()
