# Hugging Face Hub 업로드 (선택 사항)
print("📤 Hugging Face Hub 업로드")
print("=" * 50)

upload_code = '''
# Hub에 로그인 (토큰 필요)
from huggingface_hub import login
login()  # 또는 login(token="your_token")

# 모델 업로드
model.push_to_hub("your-username/grpo-math-model")
tokenizer.push_to_hub("your-username/grpo-math-model")

print("✅ 업로드 완료!")
print("모델 URL: https://huggingface.co/your-username/grpo-math-model")
'''

print("아래 코드로 Hugging Face Hub에 업로드할 수 있습니다:")
print("-" * 50)
print(upload_code)

print("\n💡 팁:")
print("  1. https://huggingface.co/settings/tokens 에서 토큰 생성")
print("  2. 'Write' 권한이 있는 토큰 사용")
print("  3. 모델 이름은 고유해야 함")
