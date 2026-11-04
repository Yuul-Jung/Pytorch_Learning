"""07차시: 외부 모델 다운로드 없이 SFT 데이터 검증·정제·분할."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path


def normalized(text):
    # 공백 차이만 있는 중복 질문도 같은 질문으로 취급한다.
    return re.sub(r"\s+", " ", text.strip()).casefold()


def validate(row):
    # 최소 필드와 데이터형 검사: 임의의 비문자열은 거부한다.
    if not isinstance(row, dict):
        return None, "객체가 아님"
    if not isinstance(row.get("instruction"), str) or not isinstance(row.get("response"), str):
        return None, "필수 문자열 누락"
    q, a = row["instruction"].strip(), row["response"].strip()
    if not q or not a:
        return None, "빈 질문/답변"
    if len(q) > 400 or len(a) > 1200:
        return None, "길이 제한 초과"
    # 실무에서는 비밀정보·개인정보 탐지 규칙을 별도로 확장해야 한다.
    return {"instruction": q, "response": a}, None


def build(input_path, out_dir, seed=42, ratio=0.2):
    if not 0 < ratio < 1:
        raise ValueError("검증 비율은 0과 1 사이여야 합니다")
    out_dir.mkdir(parents=True, exist_ok=True)
    seen, clean, rejected = set(), [], []
    with input_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                rejected.append({"line": line_no, "reason": "JSON 오류"})
                continue
            item, reason = validate(row)
            if reason:
                rejected.append({"line": line_no, "reason": reason})
                continue
            key = normalized(item["instruction"])
            if key in seen:
                rejected.append({"line": line_no, "reason": "질문 중복"})
                continue
            seen.add(key)
            clean.append(item)
    if len(clean) < 2:
        raise ValueError("유효한 고유 질문이 최소 2개 필요합니다")
    rng = random.Random(seed)
    rng.shuffle(clean)
    n_val = max(1, min(len(clean)-1, round(len(clean)*ratio)))
    val, train = clean[:n_val], clean[n_val:]
    def write(name, rows):
        path = out_dir/name
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                # SFTTrainer의 대화형 데이터 형식으로 변환한다.
                messages = [{"role": "user", "content": row["instruction"]},
                            {"role": "assistant", "content": row["response"]}]
                f.write(json.dumps({"messages": messages}, ensure_ascii=False)+"\n")
    write("train.jsonl", train)
    write("validation.jsonl", val)
    (out_dir/"rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
    train_keys = {normalized(r["instruction"]) for r in train}
    val_keys = {normalized(r["instruction"]) for r in val}
    assert train_keys.isdisjoint(val_keys), "질문 단위 데이터 누출"
    report = {"input": str(input_path), "seed": seed, "train": len(train),
              "validation": len(val), "rejected": len(rejected),
              "overlap": len(train_keys & val_keys),
              "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}
    (out_dir/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("lesson07_raw.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("processed"))
    args = parser.parse_args()
    build(args.input, args.output)