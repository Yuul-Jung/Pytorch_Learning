"""09차시: SFTTrainer + LoRA. --dry-run은 GPU 없이 데이터 형식만 검사."""
import argparse
import json
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # 교육용 모델 선택


def read_data(path):
    """JSONL의 messages 구조와 최소 user/assistant 존재를 검증한다."""
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        msgs = row.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ValueError(f"{path}:{n}: messages 누락")
        roles = []
        for msg in msgs:
            if not isinstance(msg, dict) or msg.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(f"{path}:{n}: 역할 형식 오류")
            if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                raise ValueError(f"{path}:{n}: 빈 내용")
            roles.append(msg["role"])
        if "user" not in roles or "assistant" not in roles:
            raise ValueError(f"{path}:{n}: user/assistant 쌍 필요")
        rows.append({"messages": msgs})
    if not rows:
        raise ValueError(f"{path}: 비어 있는 데이터")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--validation", default="data/validation.jsonl")
    p.add_argument("--output", default="outputs/lesson09_lora")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    train, valid = read_data(args.train), read_data(args.validation)
    # 학습/검증 사이 동일 대화가 중복되지 않도록 원문을 비교한다.
    canonical = lambda x: json.dumps(x["messages"], ensure_ascii=False, sort_keys=True)
    overlap = {canonical(x) for x in train} & {canonical(x) for x in valid}
    if overlap:
        raise ValueError(f"학습/검증 데이터 중복: {len(overlap)}건")
    print(f"DATA PASS: train={len(train)}, validation={len(valid)}, overlap=0")
    if args.dry_run:
        print("DRY RUN PASS: 모델 다운로드·GPU 학습 미실행")
        return

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다. 우선 --dry-run을 실행하세요.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Qwen2.5 attention 투영 계층만 저랭크 행렬로 학습한다.
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=args.output, max_length=256,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
        max_steps=5, learning_rate=1e-4,
        logging_steps=1, eval_strategy="steps", eval_steps=5,
        save_strategy="no", report_to="none", fp16=True,
        seed=42, assistant_only_loss=False,
    )
    trainer = SFTTrainer(
        model=MODEL_ID, args=config,
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(valid),
        processing_class=tokenizer, peft_config=lora,
    )
    trainer.model.print_trainable_parameters()
    result = trainer.train()
    print("학습 손실:", result.training_loss)
    print("검증 결과:", trainer.evaluate())
    # 여기에는 전체 기본 모델이 아닌 LoRA 어댑터가 저장된다.
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("LoRA 어댑터 저장:", args.output)


if __name__ == "__main__":
    main()