# ============================================
# 📌 train_grpo.py
# 목적: GRPOTrainer를 이용해 모델 학습 실행
# - 학습 시작/완료 로그 출력
# - 학습 결과 (스텝, Loss) 확인 가능
# - 예외 처리: 메모리 부족 시 배치 크기 조정 안내
# ============================================

# GRPOTrainer 생성 및 학습
print("\n🚀 GRPO 학습 시작")
print("=" * 50)

# Trainer 생성
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=math_reward_function,
)

# 학습 실행
print("학습 진행 중... (로그를 확인하세요)")
print("-" * 50)

try:
    train_result = trainer.train()

    print("\n" + "=" * 50)
    print("✅ 학습 완료!")
    print(f"   총 스텝: {train_result.global_step}")
    print(f"   최종 Loss: {train_result.training_loss:.4f}")

except Exception as e:
    print(f"\n⚠️ 학습 중 오류 발생: {e}")
    print("메모리 부족일 경우 배치 크기를 줄여보세요.")
