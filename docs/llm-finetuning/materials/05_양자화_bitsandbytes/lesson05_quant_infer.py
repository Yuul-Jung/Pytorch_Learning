"""05차시: 4비트 양자화 모델 로딩과 메모리 관찰 (학습이 아닌 추론 실습)."""
import argparse
import importlib.metadata as metadata
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # 교육용 선택 모델

def show_versions():
    for package in ("torch", "transformers", "accelerate", "bitsandbytes"):
        try:
            print(f"{package}: {metadata.version(package)}")
        except metadata.PackageNotFoundError:
            print(f"{package}: 설치되지 않음")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("4bit", "16bit"), default="4bit")
    args = parser.parse_args()
    show_versions()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU가 없습니다. 설치/개념 실습만 진행하세요.")
    print("GPU:", torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    options = {"device_map": "auto"}
    if args.mode == "4bit":
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        options["dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **options)
    messages = [{"role": "user", "content": "양자화를 한 문장으로 설명해 줘."}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt"
    ).to(next(model.parameters()).device)
    with torch.inference_mode():
        output = model.generate(inputs, max_new_tokens=64, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    print("응답:", tokenizer.decode(output[0, inputs.shape[-1]:],
                                  skip_special_tokens=True))
    print("CUDA peak allocated MiB:",
          round(torch.cuda.max_memory_allocated() / 1024**2, 1))
    print("CUDA peak reserved MiB:",
          round(torch.cuda.max_memory_reserved() / 1024**2, 1))
    print("주의: CUDA peak는 모델 파일 크기나 전체 시스템 RAM과 다릅니다.")

if __name__ == "__main__":
    main()