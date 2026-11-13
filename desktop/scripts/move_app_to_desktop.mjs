import { cp, lstat, mkdir, rename, rm } from "node:fs/promises";
import { homedir, platform } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const appName = "Automation Agent.app";
const desktopDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const buildOutput = path.join(
  desktopDir,
  "src-tauri",
  "target",
  "release",
  "bundle",
  "macos",
  appName,
);

if (platform() !== "darwin") {
  console.log("桌面 App 仅支持 macOS，跳过移动步骤。");
  process.exit(0);
}

try {
  await lstat(buildOutput);
} catch {
  console.error(`找不到构建产物：${buildOutput}`);
  process.exit(1);
}

const desktopPath = path.join(homedir(), "Desktop");
const destination = path.join(desktopPath, appName);
const staging = path.join(desktopPath, `.Automation Agent.${process.pid}.new.app`);
const previous = path.join(desktopPath, `.Automation Agent.${process.pid}.previous.app`);

let sourceMoved = false;
let sourceCopied = false;
let previousMoved = false;
let installed = false;

await mkdir(desktopPath, { recursive: true });
await rm(staging, { recursive: true, force: true });

try {
  try {
    await rename(buildOutput, staging);
    sourceMoved = true;
  } catch (error) {
    if (error.code !== "EXDEV") throw error;
    await cp(buildOutput, staging, { recursive: true, preserveTimestamps: true });
    sourceCopied = true;
  }

  try {
    await rename(destination, previous);
    previousMoved = true;
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }

  await rename(staging, destination);
  installed = true;
} catch (error) {
  if (!installed) {
    if (previousMoved) {
      await rename(previous, destination).catch(() => {});
    }
    if (sourceMoved) {
      await rename(staging, buildOutput).catch(() => {});
    } else {
      await rm(staging, { recursive: true, force: true }).catch(() => {});
    }
  }
  throw error;
}

if (previousMoved) {
  await rm(previous, { recursive: true, force: true }).catch((error) => {
    console.warn(`新 App 已就位，但旧版清理失败：${error.message}`);
  });
}
if (sourceCopied) {
  await rm(buildOutput, { recursive: true, force: true }).catch((error) => {
    console.warn(`新 App 已就位，但构建目录清理失败：${error.message}`);
  });
}

console.log(`已将最新 App 移到桌面：${destination}`);
