# 환경 확인
import torch
from tqdm import tqdm

print("🖥️ 실습 환경 확인")
print("=" * 50)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    if gpu_mem >= 20:
        print("\n✅ 충분한 GPU 메모리 - 기본 설정으로 진행")
        USE_SMALL_MODEL = False
    else:
        print("\n⚠️ 제한된 GPU 메모리 - 최적화 설정 적용")
        USE_SMALL_MODEL = True
else:
    print("\n⚠️ GPU 없음 - CPU 모드 (매우 느림, 데모만 권장)")
    USE_SMALL_MODEL = True
