"""桌面托管运行时的 PyInstaller 入口，从宿主接收本次启动凭证。"""

import os
import re

if __name__ == "__main__":
    # 在 runtime 导入配置和读取 .env 前取出凭证，避免配置覆盖或后续子进程继承。
    token = os.environ.pop("AUTOMATION_AGENT_SIDECAR_TOKEN", "")
    readiness = os.environ.pop("AUTOMATION_AGENT_SIDECAR_READINESS", "")
    if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in (token, readiness)):
        raise SystemExit("缺少有效的桌面运行时启动凭证，请从桌面应用启动。")

    from runtime.__main__ import main

    main(desktop_credentials=(token, readiness))
