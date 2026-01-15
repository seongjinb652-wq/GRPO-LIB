# ============================================
# 📌 memory_optimized_config.py
# 목적: GPU 메모리 크기에 따라 학습 설정 자동 최적화
# - A100, L4, T4, RTX 등 환경별 권장 파라미터 제공
# - 배치 크기, 그룹 크기, 정밀도(fp16/bf16), LoRA 사용 여부, 모델 크기 조정
# - 현재 환경을 감지하여 권장 설정 출력
# ============================================
# 📝 메모리 최적화 설정 예시
def get_optimized_config(gpu_memory_gb: float) -> dict:
    """
    GPU 메모리에 따른 최적화 설정 반환

    Args:
        gpu_memory_gb: GPU 메모리 (GB)

    Returns:
        최적화된 설정 딕셔너리
    """
    if gpu_memory_gb >= 40:  # A100 40GB+
        config = {
            "batch_size": 8,
            "num_generations": 16,
            "precision": "bf16",
            "lora": False,
            "gradient_checkpointing": False,
            "model_size": "7B",
        }
    elif gpu_memory_gb >= 20:  # L4 24GB
        config = {
            "batch_size": 4,
            "num_generations": 8,
            "precision": "bf16",
            "lora": True,
            "gradient_checkpointing": True,
            "model_size": "1.5B~3B",
        }
    elif gpu_memory_gb >= 14:  # T4 16GB
        config = {
            "batch_size": 2,
            "num_generations": 4,
            "precision": "fp16",
            "lora": True,
            "gradient_checkpointing": True,
            "model_size": "0.5B~1.5B",
        }
    else:  # 작은 GPU
        config = {
            "batch_size": 1,
            "num_generations": 2,
            "precision": "fp16",
            "lora": True,
            "gradient_checkpointing": True,
            "model_size": "0.5B",
        }

    return config

# 현재 환경에 맞는 설정 출력
print("🧠 GPU 메모리별 최적화 설정")
print("=" * 60)

gpu_options = [
    (40, "A100 40GB"),
    (24, "L4 24GB"),
    (16, "T4 16GB"),
    (8, "RTX 3070 8GB"),
]

for mem, name in gpu_options:
    config = get_optimized_config(mem)
    print(f"\n📌 {name}:")
    print(f"   배치 크기: {config['batch_size']}")
    print(f"   그룹 크기: {config['num_generations']}")
    print(f"   정밀도: {config['precision']}")
    print(f"   LoRA 사용: {'✅' if config['lora'] else '❌'}")
    print(f"   권장 모델: {config['model_size']}")

# 현재 환경
if torch.cuda.is_available():
    current_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n{'='*60}")
    print(f"🖥️ 현재 환경: {torch.cuda.get_device_name(0)} ({current_mem:.0f}GB)")
    current_config = get_optimized_config(current_mem)
    print(f"   → 권장 설정: 배치={current_config['batch_size']}, "
          f"그룹={current_config['num_generations']}, "
          f"모델={current_config['model_size']}")
