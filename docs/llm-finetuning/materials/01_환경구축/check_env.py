"""01차시: Python/PyTorch/CUDA 환경과 실습 가능 여부를 확인합니다."""
import importlib.metadata as md
import platform

print('Python:', platform.python_version())
for name in ('torch', 'transformers', 'trl', 'peft', 'datasets', 'accelerate', 'bitsandbytes'):
    try:
        print(f'{name}: {md.version(name)}')
    except md.PackageNotFoundError:
        print(f'{name}: 미설치')

try:
    import torch
    print('CUDA 사용 가능:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        print('총 VRAM(GiB):', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
        print('BF16 지원:', torch.cuda.is_bf16_supported())
except ImportError:
    print('PyTorch 미설치')