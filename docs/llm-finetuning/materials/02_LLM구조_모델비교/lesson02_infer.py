# lesson02_infer.py — 02차시 실습: 토큰화와 짧은 문장 생성
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # 교육용 선택 모델


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="FPGA가 무엇인지 두 문장으로 설명해 주세요.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    # 모델과 같은 저장소의 토크나이저를 사용한다.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()  # 평가/추론 모드: 학습용 동작 비활성화

    messages = [{"role": "user", "content": args.prompt}]
    # 채팅 템플릿이 사용자/어시스턴트 구분을 모델 형식에 맞게 변환한다.
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt"
    )
    with torch.inference_mode():  # 역전파용 계산 그래프를 만들지 않는다.
        output = model.generate(
            encoded, max_new_tokens=args.max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    new_ids = output[0, encoded.shape[-1]:]  # 입력 토큰을 제외한 생성 부분
    print("입력 토큰 수:", encoded.shape[-1])
    print("생성 토큰 수:", new_ids.shape[-1])
    print("모델 응답:", tokenizer.decode(new_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()