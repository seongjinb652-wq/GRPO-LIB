# 수학 문제 데이터셋 생성
from datasets import Dataset
import random

def generate_math_problems(num_samples: int = 200) -> Dataset:
    """
    단계별 풀이가 필요한 수학 문제 데이터셋 생성

    다양한 연산 포함:
    - 덧셈, 뺄셈
    - 곱셈
    - 간단한 방정식
    """
    data = []
    random.seed(42)

    for _ in range(num_samples):
        problem_type = random.choice(["add", "sub", "mul", "word"])

        if problem_type == "add":
            a, b = random.randint(1, 100), random.randint(1, 100)
            prompt = f"다음을 계산하세요: {a} + {b} = ?"
            answer = str(a + b)

        elif problem_type == "sub":
            a = random.randint(50, 150)
            b = random.randint(1, a)  # 음수 방지
            prompt = f"다음을 계산하세요: {a} - {b} = ?"
            answer = str(a - b)

        elif problem_type == "mul":
            a, b = random.randint(2, 12), random.randint(2, 12)
            prompt = f"다음을 계산하세요: {a} × {b} = ?"
            answer = str(a * b)

        else:  # word problem
            a = random.randint(5, 20)
            b = random.randint(1, a)
            templates = [
                (f"철수가 사과 {a}개를 가지고 있었습니다. 영희에게 {b}개를 주었습니다. 남은 사과는?", str(a - b)),
                (f"과자가 {a}개 있습니다. {b}개를 더 샀습니다. 총 과자 수는?", str(a + b)),
            ]
            prompt, answer = random.choice(templates)

        # 시스템 프롬프트 추가
        full_prompt = f"""당신은 수학 문제를 단계별로 풀어주는 도우미입니다.
<think> 태그 안에 풀이 과정을 작성하고, 마지막에 "답: [숫자]" 형식으로 답을 제시하세요.

문제: {prompt}

"""
        data.append({
            "prompt": full_prompt,
            "answer": answer,
        })

    return Dataset.from_list(data)

# 데이터셋 생성
train_dataset = generate_math_problems(200)

print("📊 데이터셋 준비 완료")
print("=" * 50)
print(f"학습 데이터: {len(train_dataset)}개")

print("\n📝 샘플 데이터:")
print("-" * 50)
sample = train_dataset[0]
print(f"프롬프트:\n{sample['prompt'][:200]}...")
print(f"\n정답: {sample['answer']}")
