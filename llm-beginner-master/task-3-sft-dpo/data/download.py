"""下载任务三所需资源：Qwen2.5-0.5B 基座模型 + MOSS SFT 数据 + MOSS plugin 数据。

DPO 偏好数据需自行选择（提示见末尾）。
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # 新增：禁用 hf_transfer
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 延长超时时间

import sys
from pathlib import Path
from huggingface_hub import snapshot_download

DATA_DIR = Path(__file__).parent
MODELS_DIR = DATA_DIR.parent / "models"


def download_base_model():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("[错误] pip install huggingface_hub")
    print("下载 Qwen/Qwen2.5-0.5B 到 models/Qwen2.5-0.5B ...")
    snapshot_download("Qwen/Qwen2.5-0.5B",
                      local_dir=str(MODELS_DIR / "Qwen2.5-0.5B"))
    print("完成")

def download_sft_data():
    """下载 MOSS-003 SFT 数据（不含工具调用版本）"""
    print("\n========== 下载 MOSS-003 SFT 数据 ==========")
    snapshot_download(
        repo_id="OpenMOSS-Team/moss-003-sft-data",
        repo_type="dataset",
        local_dir="./data/moss-sft",
        allow_patterns=["moss-003-sft-no-tools.jsonl.zip"],
        resume_download=True,
        max_workers=1
    )
    print("SFT 数据下载完成 → ./data/moss-sft")


def download_plugin_data():
    """下载 MOSS-003 插件数据（含工具调用对话，供任务五使用）"""
    print("\n========== 下载 MOSS-003 插件数据 ==========")
    snapshot_download(
        repo_id="OpenMOSS-Team/moss-003-sft-data",
        repo_type="dataset",
        local_dir="./data/moss-sft-plugin",
        allow_patterns=["moss-003-sft-with-tools-no-text2image.zip"],
        resume_download=True,
        max_workers=1
    )
    print("插件数据下载完成 → ./data/moss-sft-plugin")

    # 如需图像生成工具样例，可取消下面注释，额外下载 text2image 文件
    # snapshot_download(
    #     repo_id="OpenMOSS-Team/moss-003-sft-data",
    #     repo_type="dataset",
    #     local_dir="./data/moss-sft-plugin-text2image",
    #     allow_patterns=["moss-003-sft-with-tools-text2image.zip"],
    #     resume_download=True,
    #     max_workers=1
    # )


def download_dpo_data():
    """下载 DPO 偏好数据（默认下载中英混合版 hiyouga/DPO-En-Zh-20k）"""
    print("\n========== 下载 DPO 偏好数据 ==========")

    # 方式一：中英混合（推荐，使用 hiyouga/DPO-En-Zh-20k）
    snapshot_download(
        repo_id="hiyouga/DPO-En-Zh-20k",
        repo_type="dataset",
        local_dir="./data/dpo-en-zh-20k",
        resume_download=True,
        max_workers=1
    )
    print("DPO 数据（中英混合）下载完成 → ./data/dpo-en-zh-20k")
def hint_sft_data():
    print("\n--- MOSS-003-sft-data（SFT 数据）---")
    print("HF 数据集：https://huggingface.co/datasets/OpenMOSS-Team/moss-003-sft-data")
    print("推荐直接下载 jsonl.zip，避免 dataset viewer / 自动 builder 解析大文件失败：")
    print("  huggingface-cli download OpenMOSS-Team/moss-003-sft-data "
          "moss-003-sft-no-tools.jsonl.zip --repo-type dataset --local-dir ./data/moss-sft")
    print("GitHub：https://github.com/OpenMOSS/MOSS  (数据集 release)")


def hint_plugin_data():
    print("\n--- MOSS-003-sft-plugin（带工具调用对话，给任务五贯通用）---")
    print("同一 HF 数据集内含 with-tools 文件：OpenMOSS-Team/moss-003-sft-data")
    print("  huggingface-cli download OpenMOSS-Team/moss-003-sft-data "
          "moss-003-sft-with-tools-no-text2image.zip --repo-type dataset --local-dir ./data/moss-sft-plugin")
    print("  # 若需要图像生成工具样例，可另取 moss-003-sft-with-tools-text2image.zip")


def hint_dpo_data():
    print("\n--- DPO 偏好数据（自选）---")
    print("候选：")
    print("  - hiyouga/DPO-En-Zh-20k       中英混合")
    print("  - argilla/distilabel-intel-orca-dpo-pairs  英文")
    print("或自行用 GPT-4 / Claude 给已有 SFT 数据打偏好标签")


def main():
    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 下载慢可设 HF_ENDPOINT=https://hf-mirror.com\n")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    download_base_model()
    hint_sft_data()
    hint_plugin_data()
    hint_dpo_data()
    download_sft_data()
    download_plugin_data()
    download_dpo_data()


if __name__ == "__main__":
    main()
