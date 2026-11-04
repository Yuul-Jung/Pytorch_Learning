"""08차시: 07차시 JSONL을 활용한 SFTTrainer 기초 학습 (전체 파라미터 학습)."""
import argparse
import json
from pathlib import Path


def read_messages(path):
    """대화 데이터의 역할과 내용 형식을 검사한다."""
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}:{number}: messages 목록 누락")
        for msg in messages:
            if msg.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(f"{path}:{number}: role 오류")
            if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                raise ValueError(f"{path}:{number}: content 오류")
        records.append({"messages": messages})
    if not records:
        raise ValueError(f"{path}: 유효한 데이터가 없습니다")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="processed/train.jsonl")
    parser.add_argument("--validation", default="processed/validation.jsonl")
    parser.add_argument("--output", default="outputs/lesson08_sft")
    parser.add_argument("--dry-run", action="store_true", help="데이터만 검사하고 학습하지 않음")
    args = parser.parse_args()
    train_rows = read_messages(args.train)
    valid_rows = read_messages(args.validation)
    print(f"데이터 확인: train={len(train_rows)}, validation={len(valid_rows)}")
    if args.dry_run:
        print("DRY RUN: 데이터 검증 완료, 모델 학습 미실행")
        return

    # 학습 시에만 무거운 라이브러리를 불러온다.
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"  # 교육용 제작 선택
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not torch.cuda.is_available():
        raise RuntimeError("이 실습의 전체 파라미터 학습은 CUDA GPU 환경에서 실행하세요")
    config = SFTConfig(
        output_dir=args.output,
        max_length=256,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=5,
        learning_rate=2e-5,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="no",
        report_to="none",
        fp16=True,
        seed=42,
        assistant_only_loss=False,  # 채팅 템플릿의 assistant mask 지원 여부와 무관하게 실행
    )
    trainer = SFTTrainer(
        model=model_id,
        args=config,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(valid_rows),
        processing_class=tokenizer,
    )
    result = trainer.train()
    metrics = trainer.evaluate()
    print("학습 손실:", result.training_loss)
    print("검증 지표:", metrics)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("모델 저장:", args.output)


if __name__ == "__main__":
    main()