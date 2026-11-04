# 03차시: Hugging Face 소형 모델 추론
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # 교육용 선택 모델
prompt = "UART가 무엇인지 두 문장으로 설명해줘."
# 동일 모델의 토크나이저와 가중치를 적재한다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=True
)
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
# 프롬프트를 제외한 생성 토큰만 복원한다.
generated = outputs[0][inputs["input_ids"].shape[-1]:]
print("model:", MODEL_ID)
print("input_tokens:", inputs["input_ids"].shape[-1])
print("output_tokens:", generated.shape[-1])
print("answer:", tokenizer.decode(generated, skip_special_tokens=True))