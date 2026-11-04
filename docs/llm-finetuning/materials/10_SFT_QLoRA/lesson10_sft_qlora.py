"""10차시: 4비트 NF4 + SFTTrainer + LoRA; --dry-run은 GPU 없이 데이터 검사."""
import argparse
import json
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # 제작상 선택한 교육용 모델


def load_rows(path):
    """대화형 JSONL 데이터의 기본 구조를 검증한다."""
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        msgs = row.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise ValueError(f"{path}:{n}: messages 배열 누락")
        roles = []
        for msg in msgs:
            if not isinstance(msg, dict) or msg.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(f"{path}:{n}: role 오류")
            if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                raise ValueError(f"{path}:{n}: content 오류")
            roles.append(msg["role"])
        if "user" not in roles or "assistant" not in roles:
            raise ValueError(f"{path}:{n}: user와 assistant가 모두 필요")
        rows.append({"messages": msgs})
    if not rows:
        raise ValueError(f"{path}: 빈 데이터")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--validation", default="data/validation.jsonl")
    parser.add_argument("--output", default="outputs/lesson10_qlora")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train, valid = load_rows(args.train), load_rows(args.validation)
    canonical = lambda x: json.dumps(x["messages"], ensure_ascii=False, sort_keys=True)
    if {canonical(x) for x in train} & {canonical(x) for x in valid}:
        raise ValueError("학습·검증 데이터의 완전 동일 대화 중복")
    print(f"DATA PASS: train={len(train)}, validation={len(valid)}, overlap=0")
    if args.dry_run:
        print("DRY RUN PASS: 모델 다운로드·4비트 변환·GPU 학습 미실행")
        return

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 필요: 먼저 --dry-run으로 데이터 검사")
    # BF16 지원 GPU이면 BF16, 아니면 FP16으로 연산한다.
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    # (변경) TRL에 모델 ID와 양자화 설정을 직접 전달한다.
    # TRL/PEFT가 기본 가중치의 4비트 적재와 LoRA 학습 준비를 담당한다.
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM",
    )
    config = SFTConfig(
        output_dir=args.output, max_length=256,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
        max_steps=5, learning_rate=1e-4,
        logging_steps=1, eval_strategy="steps", eval_steps=5,
        save_strategy="no", report_to="none", seed=42,
        bf16=(compute_dtype == torch.bfloat16), fp16=(compute_dtype == torch.float16),
        gradient_checkpointing=True, assistant_only_loss=False,
    )
    trainer = SFTTrainer(
        model=MODEL_ID, args=config, train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(valid), processing_class=tokenizer,
        peft_config=lora, quantization_config=quant,
    )
    trainer.model.print_trainable_parameters()
    result = trainer.train()
    print("학습 손실:", result.training_loss)
    print("검증 결과:", trainer.evaluate())
    # 저장 대상은 기본 모델이 아닌 학습된 LoRA 어댑터이다.
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("QLoRA 어댑터 저장:", args.output)


if __name__ == "__main__":
    main()