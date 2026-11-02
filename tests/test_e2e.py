"""
端到端测试：工具、Agent 装配、工具调用链、只读闸门、记忆、沙箱边界。

主要职责：
1. 守住 agent/ 层对外的行为契约。
2. 把调试时踩过的坑固化成回归用例，坑不会二次踩。
3. 全部用假模型跑，不调真实 API、不花钱、不限流。

为什么用假模型：
- 真实模型输出不确定，断言没法写死；
- 真实模型有额度限制，全量跑一遍要花钱；
- 假模型能精确编排「每一轮该说什么」，
  出错时直接定位到第几轮，不用猜。

注意：
FakeListChatModel 不支持 tool_calls（bind_tools 未实现），
要测工具调用链必须用 GenericFakeChatModel，它接受一个 AIMessage 迭代器。
"""

from __future__ import annotations

# 标准库：asyncio 跑异步端点，sys / pathlib 处理导入路径。
import asyncio
import sys
from pathlib import Path

# pytest：测试框架，本文件用它做参数化（parametrize）。
import pytest

# 把项目根目录插到 sys.path 最前面。
# 这样不管从哪个目录跑 pytest，都能 import 到 agent / runtime 包。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 第三方库：
# GenericFakeChatModel：可编排输出的假模型，支持 tool_calls。
# AIMessage / HumanMessage / ToolMessage：agent 循环里的三种消息类型，
#   测试靠 isinstance 区分它们各自扮演的角色。
# InMemorySaver：进程内存版 Checkpointer，用完即弃，
#   不会往 data/memory.sqlite 里落东西。
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

# 项目内部：
# AgentContext：单次调用的运行时上下文，写权限 allow_write 在里面。
# build_agent：装配 Agent 的唯一入口。
# ALL_TOOLS：最终注入 create_agent 的全部工具清单。
# SANDBOX：文件工具允许触碰的根目录，也就是 data/。
from agent.context import AgentContext
from agent.graph import build_agent
from agent.tools import ALL_TOOLS
from agent.tools.files import SANDBOX


# 辅助：把「模型每一轮该说什么」编排成脚本。
#
# 真实模型不可预测，假模型才让测试有确定的断言。
# 下面这几个辅助函数就是这套编排的基础零件。


def make_model(*responses: AIMessage) -> "ToolCapableFakeModel":
    """把固定的 AIMessage 序列喂给假模型。

    必须实现 bind_tools —— BaseChatModel.bind_tools 默认抛 NotImplementedError，
    而任何带工具的 agent 都会走到这里。
    """
    return ToolCapableFakeModel(messages=iter(responses))


class ToolCapableFakeModel(GenericFakeChatModel):
    """支持 bind_tools 的假模型。

    GenericFakeChatModel 自带的 bind_tools 是空的，
    一旦 agent 走「把工具描述绑到模型上」这一步就会抛
    NotImplementedError，所以这里必须覆写。
    """

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    """构造一条「模型决定调用工具」的 AIMessage。

    type="tool_call" 不能省 —— 少了它 LangChain 认不出这是工具调用请求，
    agent 图也就不会去执行对应工具。

    call_id 是工具调用与工具结果之间的关联键：
    闸门补的 ToolMessage 要靠它对上号。
    """
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


# ---------- 1. 工具可调用性 ----------

# 守住「每个 @tool 装饰过的函数都能被正常调用」这条底线。
# 参数化跑一遍 ALL_TOOLS，任何一个工具抛异常或返回非字符串都会红。


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
def test_tool_is_callable(tool):
    """每个工具都能被 invoke，且返回字符串（而不是抛异常）。"""
    args = {
        "path": "probe.txt",
        "content": "hello",
        "data": {"a": 1},
        "pattern": "*",
        "text": "t",
        "title": "x",
        "items": "y",
    }
    result = tool.invoke(args)
    assert isinstance(result, str)


# ---------- 2. Agent 装配 ----------

# 守住「中间件栈不许被悄悄改薄」：
# 限次、重试、待办、上下文清理，外加三个自定义中间件，
# 少任何一个都会让某项能力无声失效。


def test_agent_builds_with_full_middleware_stack():
    """中间件清单保持完整：限次、重试、待办、上下文清理、自定义三者都在。"""
    import agent.graph as graph_mod

    agent = build_agent(model=make_model(AIMessage(content="ok")), checkpointer=InMemorySaver())
    assert agent is not None

    assembled = [
        *graph_mod.LIMITS,
        *graph_mod.RETRY,
        graph_mod.TODO,
        graph_mod.CONTEXT_EDITING,
        *graph_mod.CUSTOM_MIDDLEWARE,
    ]
    names = {type(m).__name__ for m in assembled}
    for expected in [
        "ModelCallLimitMiddleware",
        "ToolCallLimitMiddleware",
        "ModelRetryMiddleware",
        "ToolRetryMiddleware",
        "TodoListMiddleware",
        "ContextEditingMiddleware",
        "TaskGateMiddleware",
        "ContextInjectionMiddleware",
        "TimingMiddleware",
    ]:
        assert expected in names, f"缺少中间件 {expected}"


# ---------- 3. 工具调用链真的执行 ----------

# 守住「模型说要调工具 → 工具真的跑了 → 文件真的落盘」这条完整链路，
# 而不是模型回一句「我写好了」就结束。


def test_write_tool_actually_runs(tmp_path):
    """write_file 被真实执行，产物落在 data/ 沙箱内且内容正确。

    注意：
    tmp_path 这个 fixture 其实用不上 ——
    文件工具的沙箱在模块导入时就固定成 data/ 了，
    临时目录改不了它，所以断言要查 SANDBOX 下面的路径。
    """
    model = make_model(
        tool_call("write_file", {"path": "out/report.md", "content": "今日要点三条。"}),
        AIMessage(content="已写入。"),
    )
    agent = build_agent(model=model, checkpointer=InMemorySaver())
    result = agent.invoke(
        {"messages": [HumanMessage(content="写一份报告")]},
        {"configurable": {"thread_id": "t-write"}},
        context=AgentContext(task_name="t", allow_write=True),
    )
    assert "messages" in result
    assert (SANDBOX / "out" / "report.md").exists()
    assert "今日要点三条" in (SANDBOX / "out" / "report.md").read_text(encoding="utf-8")


# ---------- 4. 只读闸门 ----------

# 只读模式由 TaskGateMiddleware.after_model 拦截。
#
# 这里有两个坑，都要有用例守着：
# 1. 闸门漏补 ToolMessage → 历史里留着没有结果的 tool_call，
#    下一轮请求被 OpenAI / MiniMax 兼容接口直接 400
#    （tool call result does not follow tool call）。
# 2. 闸门补 ToolMessage 时如果顺手补了收尾消息，
#    agent 图的路由判定会变，模型就不再重新回答。


def test_readonly_mode_blocks_write():
    """只读模式下写操作被拦下，目标文件根本不会被创建。"""
    model = make_model(
        tool_call("write_file", {"path": "out/should_not_exist.md", "content": "x"}),
        AIMessage(content="已写入。"),
    )
    agent = build_agent(model=model, checkpointer=InMemorySaver())
    agent.invoke(
        {"messages": [HumanMessage(content="写文件")]},
        {"configurable": {"thread_id": "t-ro"}},
        context=AgentContext(task_name="ro", allow_write=False),
    )
    assert not (SANDBOX / "out" / "should_not_exist.md").exists()


def test_readonly_mode_closes_tool_calls():
    """被拦下的 tool_call 必须有对应的 ToolMessage。

    回归防护：以前只回一条 AIMessage，历史里留着没有结果的 tool_call，
    下一轮请求被 OpenAI / MiniMax 兼容接口直接 400
    （tool call result does not follow tool call）。
    """
    model = make_model(
        tool_call("write_file", {"path": "out/nope.md", "content": "x"}),
        AIMessage(content="好的。"),
    )
    agent = build_agent(model=model, checkpointer=InMemorySaver())
    result = agent.invoke(
        {"messages": [HumanMessage(content="写文件")]},
        {"configurable": {"thread_id": "t-ro-close"}},
        context=AgentContext(task_name="ro", allow_write=False),
    )

    answered = {m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)}
    pending = {
        tc["id"]
        for m in result["messages"]
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
    }
    assert pending, "用例本身没产生 tool_call，测不到东西"
    assert pending <= answered, f"这些 tool_call 没有对应的 ToolMessage：{pending - answered}"
    # 最后一条消息没有 tool_calls，agent 循环才会在闸门这里收尾
    assert not getattr(result["messages"][-1], "tool_calls", None)


# ---------- 5. 动态提示词注入 ----------

# ContextInjectionMiddleware 用 wrap_model_call 改写 system prompt。
# 守两件事：运行时上下文真的注进去了，原来的系统提示词没被冲掉。


def test_context_injected_into_system_prompt():
    """运行时上下文被拼进系统提示词，且原提示词仍在。

    为什么强调「原提示词仍在」：
    request.override() 是替换语义，
    直接把 system_message 换成新值会把原提示词整个冲掉，
    Agent 的角色设定就丢了。这条断言就是那道防线。
    """
    class Capture(ToolCapableFakeModel):
        """记录收到的消息。_generate 必须返回 ChatResult，不能返回 AIMessage。"""

        seen: list = []

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.outputs import ChatGeneration, ChatResult

            self.seen = [str(m.content) for m in messages]
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    model = Capture(messages=iter([AIMessage(content="ok")]))
    agent = build_agent(model=model, checkpointer=InMemorySaver())
    agent.invoke(
        {"messages": [HumanMessage(content="hi")]},
        {"configurable": {"thread_id": "t-prompt"}},
        context=AgentContext(task_name="prompt-test", allow_write=True),
    )
    assert model.seen, "模型没收到任何消息"
    joined = "\n".join(model.seen)
    assert "运行时上下文" in joined
    assert "prompt-test" in joined
    # override 是替换语义，这段断言守住「原系统提示词不能被冲掉」
    assert "自动化日常事务助手" in joined


# ---------- 6. 记忆 ----------

# 守住「同一 thread_id 共享历史，不同 thread_id 各自独立」。
#
# 回归防护：
# build_agent(checkpointer=None) 不是「不要记忆」，
# 而是「用默认的」—— 那会走 graph.py 的默认分支，
# 用 sqlite 落盘到 data/memory.sqlite。
# 测试会话跨次累积，最后把测试搞挂。
# 所以本文件所有用例一律显式传 InMemorySaver()。


def test_same_thread_shares_history():
    """同一 thread_id 第二次调用能看到上一轮的历史，消息条数变多。"""
    from langgraph.checkpoint.memory import InMemorySaver

    agent = build_agent(
        model=make_model(AIMessage(content="a"), AIMessage(content="b")),
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "shared"}}
    ctx = AgentContext(task_name="mem")
    r1 = agent.invoke({"messages": [HumanMessage(content="我叫张三")]}, cfg, context=ctx)
    r2 = agent.invoke({"messages": [HumanMessage(content="我叫什么")]}, cfg, context=ctx)
    assert len(r2["messages"]) > len(r1["messages"])


# ---------- 7. 中间件不因缺省 context 而崩溃 ----------

# context 是 invoke 的可选参数，不传时 runtime.context 是 None。
# 中间件里的 _ctx() 负责兜底，这条用例守住兜底没被改掉。


def test_middleware_tolerates_missing_context():
    """不传 context 时 runtime.context 是 None，中间件不能因此抛错。"""
    agent = build_agent(model=make_model(AIMessage(content="ok")), checkpointer=InMemorySaver())
    result = agent.invoke(
        {"messages": [HumanMessage(content="hi")]},
        {"configurable": {"thread_id": "t-noctx"}},
    )
    assert "messages" in result


# ---------- 8. 沙箱边界 ----------

# 文件工具的所有读写都必须落在 data/ 内。
#
# 注意这里守的是「不能写到沙箱外」这个安全属性，
# 而不是「必须抛 ValueError」这个实现细节：
# 沙箱越界往往是模型自己把路径拼错了，不是系统故障，
# 所以工具层返回一句能照做的中文说明，而不是抛异常
# （为什么不抛，见 test_sandbox_violation_is_not_retried 的说明）。


def test_file_tools_refuse_path_outside_sandbox():
    """路径越出沙箱要被拒绝，且绝不能真的写到沙箱外去。"""
    from agent.tools.files import write_file

    out = write_file.invoke({"path": "../../etc/passwd", "content": "x"})

    # 拒绝要给模型看得懂的说明。
    assert isinstance(out, str)
    assert "超出允许范围" in out
    # 说明里要带可执行的下一步，光说「不允许」模型不知道怎么改。
    assert "list_files" in out

    # 核心安全属性：沙箱外没有多出文件。
    outside = SANDBOX.parent.parent / "etc" / "passwd"
    assert not outside.exists() or outside.read_text(encoding="utf-8") != "x"


def test_sandbox_violation_is_not_retried():
    """沙箱越界不该被当成故障重试 —— 一次就返回中文说明。

    回归防护（实测踩出来的）：
        ToolRetryMiddleware 的 retry_on 默认是 (Exception,)，
        也就是所有异常都重试。越界抛 ValueError 时，
        同一个错参数会被重试 3 次、空耗约 3 秒，
        模型最后收到的还是
        「failed after 3 attempts with ValueError: ... Please try again.」
        —— 英文、没告诉它该怎么改，它很可能再拿同一个错路径试一轮。

        所以越界改成返回说明文字，断言工具消息里没有重试痕迹。
    """
    model = make_model(
        tool_call("write_file", {"path": "../../etc/passwd", "content": "x"}),
        AIMessage(content="好的。"),
    )
    agent = build_agent(model=model, checkpointer=InMemorySaver())
    result = agent.invoke(
        {"messages": [HumanMessage(content="写文件")]},
        {"configurable": {"thread_id": "t-sandbox"}},
        context=AgentContext(task_name="sandbox", allow_write=True),
    )

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs, "这次调用没有产生工具消息，用例没测到东西"

    content = tool_msgs[0].content
    assert "超出允许范围" in content
    assert "after 3 attempts" not in content, f"越界被当故障重试了：{content}"


# ---------- 8b. 读取保护：不让 agent 把大文件塞爆上下文 ----------

# 三道读取保护：大文件截断、二进制拒绝、记忆库拒绝。
# 它们返回一句说明文字而不是抛异常 ——
# 让模型换个思路，而不是把整轮推理直接打断。


def test_read_file_truncates_large_files():
    """超过上限只返回开头，并说明已截断。"""
    from agent.tools.files import MAX_READ_BYTES, read_file

    big = SANDBOX / "big_for_test.txt"
    big.write_text("A" * (MAX_READ_BYTES + 5000), encoding="utf-8")
    try:
        out = read_file.invoke({"path": "big_for_test.txt"})
        assert "已截断" in out
        assert len(out) < MAX_READ_BYTES + 1000
    finally:
        big.unlink()


def test_read_file_refuses_binary():
    """二进制文件不返回乱码，而是返回一句「这是二进制」的说明。"""
    from agent.tools.files import read_file

    blob = SANDBOX / "binary_for_test.bin"
    blob.write_bytes(b"\x00\x01\x02" + b"\xff" * 100)
    try:
        out = read_file.invoke({"path": "binary_for_test.bin"})
        assert "二进制" in out
    finally:
        blob.unlink()


def test_read_file_refuses_memory_database():
    """记忆库是运行时自己的产物，读它会撑爆上下文。"""
    from agent.tools.files import read_file

    out = read_file.invoke({"path": "memory.sqlite"})
    assert "记忆库" in out


# ---------- 8c. 工具能力补强 ----------

# 这一组守的是工具「真的好用」的几处细节。
# 每一条都对应一个实测出来的真实缺口 ——
# 缺了它们，Agent 会因为工具的缺陷而做出错误判断。


def test_list_files_recurses_into_subdirectories():
    """list_files 要能列出嵌套目录里的文件。

    回归防护：
        原来用 glob 只看顶层，data/out/report.md 这种嵌套文件根本列不出来，
        模型调 list_files("*.md") 会得到「没有匹配的文件」，
        于是误以为目录是空的。
    """
    from agent.tools.files import list_files

    nested = SANDBOX / "nested_for_test" / "deep"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "inner.md").write_text("内层文件", encoding="utf-8")
    try:
        out = list_files.invoke({"pattern": "*.md"})
        assert "nested_for_test/deep/inner.md" in out, out
    finally:
        (nested / "inner.md").unlink()
        nested.rmdir()
        (SANDBOX / "nested_for_test").rmdir()


def test_list_files_hides_memory_database():
    """list_files 不该把 11MB 的记忆库列出来误导模型。"""
    from agent.tools.files import list_files

    out = list_files.invoke({"pattern": "*"})
    assert "memory.sqlite" not in out


def test_list_files_reports_modification_time():
    """list_files 要带修改时间，模型才能判断哪些文件陈旧。

    回归防护：
        原来只返回文件名和大小，模型要回答
        「哪些文件超过 7 天没修改」时根本没有依据，只能猜。
        runtime/tasks.py 的「数据目录巡检」任务正是干这个的。
    """
    import re

    from agent.tools.files import list_files, write_file

    write_file.invoke({"path": "mtime_for_test.txt", "content": "x"})
    try:
        out = list_files.invoke({"pattern": "mtime_for_test.txt"})
        # 形如 "mtime_for_test.txt  (1 bytes, 2026-11-02 16:15)"
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", out), out
    finally:
        (SANDBOX / "mtime_for_test.txt").unlink(missing_ok=True)


def test_read_file_supports_offset_and_limit():
    """read_file 能按行分段读，并说清读到哪儿了。

    回归防护：
        原来超限提示写「需要更多内容请分段读取」，
        但根本没有分段读取的手段 —— 提示指向了一个不存在的功能。
        现在有 offset / limit，返回里还会报出总行数和下一步怎么传。
    """
    from agent.tools.files import read_file, write_file

    write_file.invoke(
        {"path": "lines_for_test.txt", "content": "\n".join(f"第{i}行" for i in range(1, 11))}
    )
    try:
        out = read_file.invoke({"path": "lines_for_test.txt", "offset": 0, "limit": 3})
        assert "第1行" in out and "第3行" in out
        assert "第4行" not in out, "limit 没生效"
        # 必须告诉模型总共多少行、下一段从哪开始。
        assert "共 10 行" in out
        assert "offset=3" in out

        # 接着读下一段。
        out2 = read_file.invoke({"path": "lines_for_test.txt", "offset": 3, "limit": 2})
        assert "第4行" in out2 and "第5行" in out2
        assert "第6行" not in out2
    finally:
        (SANDBOX / "lines_for_test.txt").unlink(missing_ok=True)


def test_save_json_accepts_array():
    """save_json 除了对象也要能存数组。

    回归防护：
        原来 data 标注成 dict，模型存 JSON 数组（极常见的数据形状）
        会触发 Pydantic 校验失败，被 ToolRetryMiddleware 重试两次
        同样的参数后彻底失败。
    """
    from agent.tools.files import read_json, save_json

    out = save_json.invoke({"path": "arr_for_test.json", "data": [1, 2, 3]})
    assert "已保存 JSON" in out
    try:
        assert read_json.invoke({"path": "arr_for_test.json"}).strip() == "[\n  1,\n  2,\n  3\n]"
    finally:
        (SANDBOX / "arr_for_test.json").unlink(missing_ok=True)


def test_read_json_reports_parse_error():
    """read_json 解析失败要说出错在哪，而不是把原文丢回去让模型自己猜。"""
    from agent.tools.files import read_json, write_file

    write_file.invoke({"path": "bad_for_test.json", "content": '{"a": 1,,,}'})
    try:
        out = read_json.invoke({"path": "bad_for_test.json"})
        assert "不是合法的 JSON" in out
        assert "第 1 行" in out, f"没给出行号，模型不知道错在哪：{out}"
    finally:
        (SANDBOX / "bad_for_test.json").unlink(missing_ok=True)


def test_write_file_leaves_no_temp_file():
    """原子写完成后不留临时文件，也不留半截文件。

    回归防护：
        直接 write_text 在写到一半被中断时会留下半截文件，
        Agent 下次读到会当成正常内容，基于残缺数据往下推理。
        现在先写 .tmp 再 os.replace，失败时清掉 .tmp。
    """
    from agent.tools.files import write_file

    out = write_file.invoke({"path": "atomic_for_test.txt", "content": "完整内容"})
    assert "已写入" in out
    try:
        assert (SANDBOX / "atomic_for_test.txt").read_text(encoding="utf-8") == "完整内容"
        # 目录里不能有 .tmp 残留。
        leftovers = [p.name for p in SANDBOX.glob("*.tmp")]
        assert not leftovers, f"留下临时文件：{leftovers}"
    finally:
        (SANDBOX / "atomic_for_test.txt").unlink(missing_ok=True)


# ---------- 9. 流式接口的会话隔离 ----------

# /run/stream 默认每次都是新会话（thread_id 随机生成），
# 只有显式传 thread_id 才复用 —— 否则所有流式请求共享同一份记忆。


def test_stream_endpoint_isolates_threads(monkeypatch):
    """流式接口默认每次新会话；显式传 thread_id 才复用。

    回归防护：曾经硬编码成 "stream-session"，导致所有流式请求共享同一份记忆。
    """
    from runtime import server
    from runtime.server import TaskRequest

    seen: list[str] = []

    class FakeAgent:
        def stream(self, messages, config, context=None, stream_mode=None):
            seen.append(config["configurable"]["thread_id"])
            yield {"messages": [AIMessage(content="ok")]}

    monkeypatch.setattr(server, "get_agent", lambda: FakeAgent())

    async def drain(resp):
        return [chunk async for chunk in resp.body_iterator]

    asyncio.run(drain(server.run_stream(TaskRequest(instruction="第一次"))))
    asyncio.run(drain(server.run_stream(TaskRequest(instruction="第二次"))))
    asyncio.run(drain(server.run_stream(TaskRequest(instruction="第三次", thread_id="fixed"))))

    assert seen[0] != seen[1], "默认不该共享会话"
    assert seen[0].startswith("stream-")
    assert seen[1].startswith("stream-")
    assert seen[2] == "fixed", "显式传的 thread_id 要被尊重"
