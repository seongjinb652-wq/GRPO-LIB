# ============================================
# 📌 run_grpo_pipeline.py
# 목적: GRPO 학습 전체 파이프라인 실행
# - 데이터셋 생성
# - 보상 함수 정의
# - 학습 설정 로드
# - 모델 로드 및 LoRA 적용
# - GRPOTrainer 학습 실행
# - 모델 저장 및 Hugging Face 업로드
# - 추론 응답 테스트
# ============================================

from generate_math_dataset import generate_math_problems
from math_reward import math_reward_function
from grpo_config import training_args
from load_model_with_lora import model, tokenizer
from train_grpo import GRPOTrainer
from model_saver import save_model_and_tokenizer
from huggingface_upload import upload_to_hub
from inference_response import generate_response

print("\n🚀 GRPO 파이프라인 시작")
print("=" * 50)

# 0. 폰트 설정. 
# 
# 1. 데이터셋 준비
train_dataset = generate_math_problems(200)

# 2. Trainer 생성
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=math_reward_function,
)

# 3. 학습 실행
train_result = trainer.train()
print(f"✅ 학습 완료: step={train_result.global_step}, loss={train_result.training_loss:.4f}")

# 4. 모델 저장
save_model_and_tokenizer(trainer, tokenizer, "./grpo_math_model")

# 5. Hugging Face Hub 업로드 (선택)
# upload_to_hub("./grpo_math_model")

# 6. 추론 응답 테스트
sample_prompt = "다음을 계산하세요: 12 + 7 = ?"
response = generate_response(model, tokenizer, sample_prompt)
print("\n📝 추론 응답 예시:")
print(response)
