# 📝 GSM8K 스타일 수학 데이터셋 생성
from datasets import Dataset
import random

def generate_math_dataset(num_samples: int = 100) -> Dataset:
    """
    간단한 수학 문제 데이터셋 생성

    GRPO 학습용 형식으로 생성
    """
    data = []

    operations = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("×", lambda a, b: a * b),
    ]

    for _ in range(num_samples):
        # 랜덤 숫자 생성
        a = random.randint(1, 50)
        b = random.randint(1, 50)

        # 랜덤 연산 선택
        op_symbol, op_func = random.choice(operations)

        # 뺄셈의 경우 음수 방지
        if op_symbol == "-" and a < b:
            a, b = b, a

        # 문제와 정답 생성
        prompt = f"다음 수학 문제를 단계별로 풀어주세요.\n\n문제: {a} {op_symbol} {b} = ?\n\n"
        answer = str(op_func(a, b))

        data.append({
            "prompt": prompt,
            "answer": answer,
        })

    return Dataset.from_list(data)

# 데이터셋 생성 및 확인
print("📊 수학 문제 데이터셋 생성")
print("=" * 60)

train_dataset = generate_math_dataset(100)

print(f"데이터셋 크기: {len(train_dataset)}개")
print("\n샘플 데이터 (3개):")
print("-" * 60)

for i in range(3):
    sample = train_dataset[i]
    print(f"\n[샘플 {i+1}]")
    print(f"프롬프트: {sample['prompt'].strip()}")
    print(f"정답: {sample['answer']}")
