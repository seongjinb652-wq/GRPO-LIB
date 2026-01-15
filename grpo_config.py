# GRPO 학습 설정
from trl import GRPOConfig, GRPOTrainer

print("🔧 GRPO 학습 설정 중...")
print("=" * 50)

# GRPOConfig 설정
training_args = GRPOConfig(
    output_dir="./grpo_math_model",

    # 학습 파라미터
    max_steps=CONFIG["max_steps"],
    per_device_train_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=4,
    learning_rate=CONFIG["learning_rate"],
    warmup_ratio=0.1,

    # GRPO 특화 파라미터
    num_generations=CONFIG["num_generations"],
    max_completion_length=256,
    max_prompt_length=256,
    temperature=0.9,
    beta=0.04,

    # 로깅 및 저장
    logging_steps=5,
    save_steps=25,
    save_total_limit=2,

    # 메모리 최적화
    bf16=(CONFIG["precision"] == "bf16"),
    fp16=(CONFIG["precision"] == "fp16"),
    gradient_checkpointing=True,
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",

    # 기타
    remove_unused_columns=False,
    seed=42,
)

print("✅ GRPOConfig 설정 완료")
print(f"   max_steps: {training_args.max_steps}")
print(f"   batch_size: {training_args.per_device_train_batch_size}")
print(f"   num_generations: {training_args.num_generations}")
