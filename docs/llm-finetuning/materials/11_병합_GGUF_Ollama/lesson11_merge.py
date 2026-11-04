#!/usr/bin/env python3
"""11차시: 학습된 LoRA 어댑터를 기본 모델에 병합한다."""
import argparse
import json
from pathlib import Path

BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def check_adapter(path: Path):
    """학습된 어댑터 설정과 가중치가 실제 존재하는지 확인한다."""
    cfg = path / "adapter_config.json"
    weights = [path / "adapter_model.safetensors", path / "adapter_model.bin"]
    if not cfg.is_file() or not any(p.is_file() for p in weights):
        raise FileNotFoundError("학습된 adapter_config.json 및 가중치가 필요합니다.")
    data = json.loads(cfg.read_text(encoding="utf-8"))
    model_id = data.get("base_model_name_or_path")
    if model_id != BASE_ID:
        raise ValueError(f"기본 모델 불일치: {model_id!r} != {BASE_ID!r}")
    return data

def merge(adapter: Path, output: Path):
    """양자화되지 않은 기본 모델에 어댑터를 병합하여 저장한다."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # 4비트로 로드한 가중치에 직접 병합하지 않는다.
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, dtype=dtype, device_map="cpu"
    )
    tuned = PeftModel.from_pretrained(base, str(adapter))
    merged = tuned.merge_and_unload(safe_merge=True)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    tokenizer.save_pretrained(str(output))
    print(f"MERGE SAVED: {output}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="../09_SFT_LoRA/outputs/lesson09_lora")
    parser.add_argument("--output", default="merged_model")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    adapter = Path(args.adapter)
    check_adapter(adapter)
    print("ADAPTER PASS: config/weights/base model match")
    if args.check_only:
        print("DRY RUN PASS: 모델 로드·병합 미실행")
        return
    merge(adapter, Path(args.output))

if __name__ == "__main__":
    main()