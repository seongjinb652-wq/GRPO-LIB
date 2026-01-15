# 학습 설정 (GPU 메모리에 따라 자동 조정)
if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    if gpu_mem >= 20:  # L4 24GB 이상
        CONFIG = {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "batch_size": 2,
            "num_generations": 4,
            "max_steps": 50,  # 데모용 짧은 학습
            "learning_rate": 5e-6,
            "use_lora": True,
            "precision": "bf16",
        }
    elif gpu_mem >= 14:  # T4 16GB
        CONFIG = {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "batch_size": 1,
            "num_generations": 2,
            "max_steps": 30,
            "learning_rate": 5e-6,
            "use_lora": True,
            "precision": "fp16",
        }
    else:  # 작은 GPU
        CONFIG = {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "batch_size": 1,
            "num_generations": 2,
            "max_steps": 20,
            "learning_rate": 5e-6,
            "use_lora": True,
            "precision": "fp16",
        }
else:  # CPU
    CONFIG = {
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "batch_size": 1,
        "num_generations": 2,
        "max_steps": 5,  # CPU에서는 최소한만
        "learning_rate": 5e-6,
        "use_lora": True,
        "precision": "fp32",
    }

print("⚙️ 학습 설정")
print("=" * 50)
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
