"""06차시: 기준 모델과 학습 완료된 LoRA 어댑터의 공정한 추론 비교."""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # 교육용 선택 모델


def load_questions(path):
    with open(path, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or not {"id", "question", "reference"}.issubset(rows[0]):
        raise ValueError("CSV에는 id, question, reference 열과 1개 이상의 행이 필요합니다.")
    return rows


def generate(model, tokenizer, question, max_new_tokens):
    messages = [{"role": "user", "content": question}]
    # (변경) 템플릿을 바로 토큰화하여 BOS/EOS 중복 삽입을 방지한다.
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="lesson06_eval.csv")
    parser.add_argument("--adapter", default=None,
                        help="학습 완료된 PEFT LoRA 어댑터 폴더; 생략하면 기준 모델만 실행")
    parser.add_argument("--output", default="lesson06_results.json")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    questions = load_questions(args.csv)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # CPU 환경에서도 동작하도록 dtype/device 선택. GPU 모델 메모리 상황 확인 필수.
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    records = []
    for row in questions:
        answer, seconds = generate(model, tokenizer, row["question"], args.max_new_tokens)
        records.append({**row, "variant": "base", "answer": answer,
                        "seconds": round(seconds, 4)})
    if args.adapter:
        from peft import PeftModel
        if not Path(args.adapter).is_dir():
            raise FileNotFoundError("학습된 LoRA 어댑터 폴더를 찾을 수 없습니다.")
        tuned = PeftModel.from_pretrained(model, args.adapter).eval()
        for row in questions:
            answer, seconds = generate(tuned, tokenizer, row["question"], args.max_new_tokens)
            records.append({**row, "variant": "adapter", "answer": answer,
                            "seconds": round(seconds, 4)})
    Path(args.output).write_text(json.dumps(records, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"결과 저장: {args.output}, 레코드: {len(records)}")
    if not args.adapter:
        print("주의: 학습 어댑터 미지정. 학습 전후 성능 비교가 아니라 기준선만 기록했습니다.")


if __name__ == "__main__":
    main()