"""文件发现工具的沙箱、过滤、截断和结果格式测试。

全部用 pytest 临时目录替换 SANDBOX，不读取开发机真实项目或凭据。重点覆盖
目录树、文件名搜索和正文搜索共用的安全边界，防止三个入口逐渐出现不同规则。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.tools import files as files_mod


# ============================================================
# 1. 临时工作区
# ============================================================

@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """建立含源码、生成目录、敏感文件、大文件、二进制和越界链接的工作区。"""
    monkeypatch.setattr(files_mod, "SANDBOX", tmp_path.resolve())

    source = tmp_path / "src"
    source.mkdir()
    (source / "Main.PY").write_text(
        "第一行\nNeedle 在这里\n" + "很长" * 200,
        encoding="utf-8",
    )
    nested = source / "nested"
    nested.mkdir()
    (nested / "helper.py").write_text("needle helper\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("needle readme\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("KEY=replace-me\n", encoding="utf-8")
    (tmp_path / ".env").write_text("needle=real-secret\n", encoding="utf-8")

    for dirname in (".hidden", ".git", "node_modules", "build"):
        target = tmp_path / dirname / "ignored.py"
        target.parent.mkdir()
        target.write_text("needle hidden\n", encoding="utf-8")

    (tmp_path / "binary.bin").write_bytes(b"\x00needle")
    (tmp_path / "large.txt").write_bytes(
        b"x" * files_mod.MAX_READ_BYTES + b"needle-after-limit"
    )

    outside = tmp_path.parent / f"{tmp_path.name}-outside-secret.txt"
    outside.write_text("needle outside\n", encoding="utf-8")
    (source / "outside-link.txt").symlink_to(outside)

    try:
        yield tmp_path
    finally:
        outside.unlink(missing_ok=True)


# ============================================================
# 2. 目录树与文件名搜索
# ============================================================

def test_directory_tree_respects_depth_filters_and_symlink_boundary(workspace):
    """目录树展示层级，同时隐藏凭据、隐藏/依赖目录和越界链接。"""
    shallow = files_mod.directory_tree.invoke({"max_depth": 1})
    assert "[D] ./" in shallow
    assert "[D] src/" in shallow
    assert "Main.PY" not in shallow
    assert ".env.example" in shallow

    tree = files_mod.directory_tree.invoke({"max_depth": 3})
    assert "[F] Main.PY" in tree
    assert "[D] nested/" in tree
    assert "[F] helper.py" in tree
    assert ".env  " not in tree and "real-secret" not in tree
    assert "outside-link.txt" not in tree and "outside-secret" not in tree
    for dirname in (".hidden", ".git", "node_modules", "build"):
        assert f"[D] {dirname}/" not in tree
        assert dirname in tree  # 过滤说明明确告知结果不是全量扫描。
    assert "已隐藏敏感配置或密钥文件" in tree


def test_directory_tree_validates_root_limits_and_truncates(workspace):
    """无效目录或无限制参数返回说明，达到条目上限时明确标记。"""
    assert "不是目录" in files_mod.directory_tree.invoke({"path": "README.md"})
    assert "1-8" in files_mod.directory_tree.invoke({"max_depth": 0})
    assert "1-500" in files_mod.directory_tree.invoke({"max_entries": 999})
    result = files_mod.directory_tree.invoke({"max_entries": 2})
    assert "已达到 2 项上限" in result


def test_search_files_matches_relative_names_without_exposing_filtered_paths(workspace):
    """文件名搜索不区分大小写，可限定目录，并沿用所有过滤规则。"""
    result = files_mod.search_files.invoke({"query": "main.py"})
    assert "src/Main.PY" in result
    assert "bytes" in result

    scoped = files_mod.search_files.invoke({"query": ".py", "path": "src"})
    assert "src/Main.PY" in scoped and "src/nested/helper.py" in scoped
    assert "ignored.py" not in scoped and "outside-link.txt" not in scoped
    assert "query 不能为空" in files_mod.search_files.invoke({"query": "  "})
    assert "1-200" in files_mod.search_files.invoke({"query": "py", "max_results": 0})


# ============================================================
# 3. 正文搜索
# ============================================================

def test_search_text_returns_line_numbers_and_applies_all_filters(workspace):
    """正文搜索返回可定位结果，不读取凭据、生成目录、二进制或大文件尾部。"""
    result = files_mod.search_text.invoke({"query": "needle"})
    assert "src/Main.PY:2: Needle 在这里" in result
    assert "src/nested/helper.py:1: needle helper" in result
    assert "README.md:1: needle readme" in result
    assert ".env:" not in result and "real-secret" not in result
    assert "ignored.py" not in result and "outside" not in result
    assert "needle-after-limit" not in result
    assert "大文件只搜索了前" in result
    assert "已跳过 1 个二进制文件" in result


def test_search_text_supports_pattern_case_scope_preview_and_result_limit(workspace):
    """范围、glob、大小写、单行预览和结果条数都按调用参数生效。"""
    py_only = files_mod.search_text.invoke({
        "query": "Needle", "path": "src", "pattern": "*.py", "case_sensitive": True,
    })
    assert "src/Main.PY" not in py_only  # glob 本身区分扩展名大小写。
    assert "没有" in py_only

    insensitive = files_mod.search_text.invoke({
        "query": "NEEDLE", "path": "src", "pattern": "*", "max_results": 1,
    })
    assert "src/Main.PY:2" in insensitive
    assert "只返回前 1 处" in insensitive

    long_line = files_mod.search_text.invoke({"query": "很长"})
    preview = next(line for line in long_line.splitlines() if "src/Main.PY:3:" in line)
    assert preview.endswith("…")
    assert len(preview) < files_mod.MAX_SEARCH_LINE_CHARS + 40


def test_search_tools_reject_sensitive_and_out_of_sandbox_roots(workspace):
    """直接把敏感路径或工作区外路径设为扫描起点也不能绕过保护。"""
    assert "安全原因" in files_mod.search_text.invoke({"query": "x", "path": ".env"})
    assert "允许范围" in files_mod.search_files.invoke({"query": "x", "path": "../"})
    assert "query 不能为空" in files_mod.search_text.invoke({"query": ""})
    assert "1-200" in files_mod.search_text.invoke({"query": "x", "max_results": 201})


def test_list_files_enforces_its_documented_result_ceiling(workspace):
    """旧列表工具也不能通过调用参数绕过 200 条输出硬上限。"""
    assert "1-200" in files_mod.list_files.invoke({"max_results": 201})
