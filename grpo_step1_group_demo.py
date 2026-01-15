# ============================================
# 📌 grpo_step1_group_demo.py
# 목적: GRPO Step 1 그룹 생성 시뮬레이션
# - 실제 GRPO에서는 LLM이 temperature sampling으로 다양한 응답을 생성
# - 여기서는 정답/오답을 랜덤하게 섞어 그룹을 구성하는 시뮬레이션
# - 교육용/개념 설명용 예시 코드로, 실제 학습에서는 Trainer 내부에서 처리됨
# ============================================
# 📝 GRPO Step 1: 그룹 생성 시뮬레이션
import numpy as np
from typing import List, Dict

def generate_response_group(prompt: str, group_size: int = 8) -> List[Dict]:
    """
    GRPO의 그룹 생성 단계 시뮬레이션

    실제로는 LLM이 temperature sampling으로 다양한 응답 생성
    여기서는 시뮬레이션으로 대체
    """
    # 시뮬레이션: 다양한 품질의 응답 생성
    responses = []

    # 정답과 오답의 비율을 랜덤하게 설정
    for i in range(group_size):
        is_correct = np.random.random() > 0.3  # 70% 확률로 정답

        if is_correct:
            # 정답 응답 (다양한 설명 방식)
            explanations = [
                "2와 3을 더하면 5가 됩니다.",
                "2 + 3을 계산하면 5입니다.",
                "먼저 2에서 시작해서 3을 더하면 5",
            ]
            thought = np.random.choice(explanations)
            answer = "5"
        else:
            # 오답 응답
            wrong_answers = ["4", "6", "7"]
            thought = "대충 계산하면..."
            answer = np.random.choice(wrong_answers)

        responses.append({
            "id": i + 1,
            "thought": thought,
            "answer": answer,
            "full_output": f"<think>{thought}</think>\n{answer}"
        })

    return responses

# 그룹 생성 시연
np.random.seed(42)  # 재현성을 위한 시드 설정
prompt = "2 + 3 = ?"
group = generate_response_group(prompt, group_size=8)

print("📦 GRPO Step 1: 그룹 생성")
print("=" * 60)
print(f"프롬프트: {prompt}")
print(f"그룹 크기: {len(group)}개 응답")
print("-" * 60)

for resp in group:
    status = "✅" if resp["answer"] == "5" else "❌"
    print(f"응답 {resp['id']}: {resp['thought'][:30]}... → {resp['answer']} {status}")
