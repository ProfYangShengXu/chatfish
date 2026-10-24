#!/usr/bin/env python3
"""下载两个模型到 ``models_dir``（默认 ``<repo>/models``）。

幂等：已存在就跳过，打印目标路径。

⚠️ 只下这两个。规格第〇节明确：
   **不用任何 >1GB 的模型**（bge-reranker-base ≈1.1GB 是唯一例外，因为它是精排必需）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from memory_bridge.config import load_config, load_env_file  # noqa: E402

MODELS: tuple[tuple[str, str, str], ...] = (
    ("BAAI/bge-small-zh-v1.5", "bge-small-zh-v1.5", "~95MB"),
    ("BAAI/bge-reranker-base", "bge-reranker-base", "~1.1GB"),
)

# 判定「已下载完整」需要的文件
_REQUIRED_FILES = ("config.json",)


def _is_complete(target: Path, *, allow_weights: bool = True) -> bool:
    if not target.is_dir():
        return False
    for name in _REQUIRED_FILES:
        if not (target / name).exists():
            return False
    if not allow_weights:
        return True
    patterns = ("*.safetensors", "pytorch_model*.bin")
    return any(any(target.glob(pattern)) for pattern in patterns)


def download(repo_id: str, target: Path) -> None:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        # 跳过用不到的大文件（onnx / tensorflow / flax 权重）
        ignore_patterns=["*.onnx", "*.h5", "*.msgpack", "*.tflite", "onnx/*"],
        max_workers=4,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 Memory Bridge 需要的两个模型")
    parser.add_argument("--models-dir", default=None, help="目标目录（默认为配置里的 models_dir）")
    parser.add_argument("--force", action="store_true", help="即使已存在也重新下载")
    args = parser.parse_args(argv)

    load_env_file(REPO_ROOT / ".env")
    cfg = load_config()
    models_dir = Path(args.models_dir) if args.models_dir else cfg.models_dir

    # 国内直连慢；除非已经显式指定了别的端点，否则走镜像。
    # ⚠️ 不修改用户已有的 HF_ENDPOINT（那是用户的选择）。
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # ⚠️ 必须关掉 Xet：新版 huggingface_hub 默认走 Xet 协议，
    #    经镜像下载时会重定向到 cas-server.xethub.hf.co 并返回
    #    **401 Unauthorized**（实测，不是「慢」）。回退到传统 HTTP 才能下。
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"models_dir = {models_dir}")
    print(f"HF_ENDPOINT = {os.environ['HF_ENDPOINT']}")

    exit_code = 0
    for repo_id, dirname, size in MODELS:
        target = models_dir / dirname
        if _is_complete(target) and not args.force:
            print(f"[skip]  {repo_id} 已存在 → {target}")
            continue
        print(f"[fetch] {repo_id} ({size}) → {target}")
        try:
            download(repo_id, target)
        except Exception as exc:  # noqa: BLE001 - 网络失败要原样报出来
            print(f"[fail]  {repo_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        print(f"[done]  {repo_id} → {target}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
