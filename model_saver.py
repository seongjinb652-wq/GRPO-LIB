# 모델 저장
print("💾 모델 저장 중...")
print("=" * 50)

save_path = "./grpo_math_model_final"

# LoRA 어댑터 저장
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ 모델 저장 완료: {save_path}")

# 저장된 파일 확인
import os
saved_files = os.listdir(save_path)
print(f"\n저장된 파일:")
for f in saved_files[:10]:  # 처음 10개만 표시
    print(f"  - {f}")
