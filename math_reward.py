# GRPO 보상 함수 정의
import re
from typing import List

def math_reward_function(
    completions: List[str],
    prompts: List[str],
    answer: List[str],
    **kwargs
) -> List[float]:
    """
    수학 문제 보상 함수

    점수 구성:
    - 정답: 0.7
    - <think> 형식 사용: 0.2
    - 적절한 길이: 0.1
    """
    rewards = []

    for completion, correct_answer in zip(completions, answer):
        reward = 0.0

        # 1. 정답 확인 (0.7점)
        # 응답에서 숫자 추출
        numbers = re.findall(r'-?\d+', completion)
        if numbers:
            # "답:" 이후의 숫자 우선, 없으면 마지막 숫자
            answer_match = re.search(r'답[:\s]*(-?\d+)', completion)
            if answer_match:
                model_answer = answer_match.group(1)
            else:
                model_answer = numbers[-1]

            if model_answer == correct_answer:
                reward += 0.7

        # 2. 형식 확인 (0.2점)
        if "<think>" in completion and "</think>" in completion:
            think_match = re.search(r'<think>(.*?)</think>', completion, re.DOTALL)
            if think_match and len(think_match.group(1).strip()) > 10:
                reward += 0.2  # 의미 있는 사고 과정
            else:
                reward += 0.1  # 형식만 맞음

        # 3. 길이 보상 (0.1점)
        length = len(completion)
        if 30 <= length <= 300:
            reward += 0.1
        elif length > 300:
            reward += 0.05  # 너무 김

        rewards.append(reward)

    return rewards

# 보상 함수 테스트
print("🎯 보상 함수 테스트")
print("=" * 50)

test_completions = [
    "<think>3 + 5를 계산합니다. 3 + 5 = 8입니다.</think>\n답: 8",
    "8",
    "<think>음...</think>\n7",
    "<think>3과 5를 더하면 8이 됩니다. 왜냐하면 3에서 5를 더하면 8이기 때문입니다.</think>\n\n답: 8",
]
test_answers = ["8", "8", "8", "8"]

rewards = math_reward_function(test_completions, [""] * 4, test_answers)

print("테스트 결과:")
for i, (comp, reward) in enumerate(zip(test_completions, rewards)):
    print(f"\n[{i+1}] 보상: {reward:.2f}")
    print(f"    응답: {comp[:60]}...")
