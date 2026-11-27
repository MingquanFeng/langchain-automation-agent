/**
 * 桌面子项目的只读 JavaScript 语法检查。
 *
 * 检查 Node 脚本和两个 HTML 页面里的内嵌脚本，不启动 Tauri、不构建 sidecar，
 * 也不执行页面代码，适合由受控 `npm run check` 白名单调用。
 */

import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const nodeScripts = ["scripts/move_app_to_desktop.mjs", "scripts/check.mjs"];
const htmlFiles = ["web/index.html", "web/workbench.html"];

for (const relativePath of nodeScripts) {
  const absolutePath = path.join(projectRoot, relativePath);
  const result = spawnSync(process.execPath, ["--check", absolutePath], { encoding: "utf8" });
  if (result.error || result.status !== 0) {
    process.stderr.write(result.stderr || String(result.error));
    process.exitCode = 1;
    continue;
  }
  console.log(`语法检查通过：${relativePath}`);
}

for (const relativePath of htmlFiles) {
  const source = await readFile(path.join(projectRoot, relativePath), "utf8");
  const inlineScripts = [...source.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script\s*>/gi)];
  inlineScripts.forEach((match, index) => {
    new vm.Script(match[1], { filename: `${relativePath}#script-${index + 1}` });
  });
  console.log(`内嵌脚本语法检查通过：${relativePath}（${inlineScripts.length} 个脚本）`);
}
