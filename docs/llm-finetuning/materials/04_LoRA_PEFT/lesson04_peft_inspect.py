# lesson04_peft_inspect.py — 04차시 LoRA 적용 및 학습 매개변수 점검
# 설치: python -m pip install "transformers>=4.45,<5" "peft>=0.13" torch
# 실행: python lesson04_peft_inspect.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # 교육용 선택 모델

# 학습 가능한 LoRA 가중치를 확인하기 위한 CPU/GPU 공통 예제
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=False
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Qwen2.5 계열 attention 투영층에 저랭크 행렬을 삽입
config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none", task_type=TaskType.CAUSAL_LM,
)
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()

# 실제 이름과 requires_grad로 학습 대상 확인
trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in peft_model.parameters())
print(f"trainable={trainable:,} total={total:,} ratio={100*trainable/total:.4f}%")
print("tokenizer vocab size:", len(tokenizer))