"""12차시: 동일 평가셋을 사용한 기준 모델/LoRA/배포 모델 비교."""
import argparse
import csv
import json
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # 교육용 선택 모델


def read_cases(path):
    """평가 CSV를 읽고 필수 필드 및 중복 질문을 확인한다."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("평가 데이터가 비어 있습니다")
    seen = set()
    for row in rows:
        if not row.get("question") or not row.get("reference"):
            raise ValueError("question/reference 값이 필요합니다")
        if row["question"] in seen:
            raise ValueError("평가 질문 중복")
        seen.add(row["question"])
    return rows


def score_exact(answer, reference):
    """공백을 정규화한 정확 일치 지표. 의미 정확도를 대신하지 않는다."""
    norm = lambda s: " ".join(s.strip().split()).casefold()
    return int(norm(answer) == norm(reference))


def summarize(records):
    """모델별 정확 일치율과 평균 지연시간을 계산한다."""
    result = {}
    for name in sorted({r["model"] for r in records}):
        group = [r for r in records if r["model"] == name]
        result[name] = {
            "count": len(group),
            "exact_match": sum(r["exact_match"] for r in group) / len(group),
            "avg_seconds": sum(r["seconds"] for r in group) / len(group),
        }
    return result


def load_hf(model_id, adapter=None):
    """실제 실행 모드에서만 모델과 토크나이저를 내려받는다."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", device_map="auto"
    )
    if adapter:
        from peft import PeftModel
        if not (Path(adapter) / "adapter_config.json").exists():
            raise FileNotFoundError("학습된 어댑터가 없습니다: " + adapter)
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def generate_hf(model, tokenizer, question):
    """동일 생성 조건을 적용하고 새로 생성된 토큰만 디코딩한다."""
    import torch
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True
    ).to(model.device)
    # (추가) CUDA 비동기 실행 시간을 정확히 측정하기 위해 시작 전 동기화한다.
    if torch.cuda.is_available() and str(model.device).startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    if torch.cuda.is_available() and str(model.device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(Path(__file__).with_name("lesson12_eval.csv")))
    parser.add_argument("--adapter", default="")
    parser.add_argument("--output", default="lesson12_results.json")
    parser.add_argument("--run-models", action="store_true", help="실제 모델 추론 수행")
    args = parser.parse_args()
    cases = read_cases(args.data)
    if not args.run_models:
        print(f"DRY RUN PASS: {len(cases)}개 평가 문항 확인, GPU/모델 실행 없음")
        return
    if not args.adapter:
        raise ValueError("전후 비교를 위해 --adapter 학습 결과 경로가 필요합니다")
    records = []
    for name, adapter in [("base", None), ("lora", args.adapter)]:
        model, tokenizer = load_hf(MODEL_ID, adapter)
        for case in cases:
            answer, seconds = generate_hf(model, tokenizer, case["question"])
            records.append({"model": name, "question": case["question"],
                            "reference": case["reference"], "answer": answer,
                            "seconds": round(seconds, 4),
                            "exact_match": score_exact(answer, case["reference"])})
        del model, tokenizer
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {"model_id": MODEL_ID, "settings": {"max_new_tokens": 64,
               "do_sample": False}, "summary": summarize(records), "records": records}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()