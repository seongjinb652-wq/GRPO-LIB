# ============================================
# 📌 grpo_step2_rewards_demo..py
# 목적: GRPO Step 2 보상 계산 및 Advantage 정규화 시뮬레이션
# - 각 응답에 대해 정답 여부로 보상(1.0/0.0) 계산
# - 그룹 평균(μ)과 표준편차(σ)를 이용해 Advantage 정규화
# - Advantage 값에 따라 확률 증가/감소/변화 없음 액션 표시
# - 실제 학습에서는 GRPOTrainer 내부에서 자동 처리되며,
#   여기서는 개념 설명용 예시 코드
# ============================================
# 📝 GRPO Step 2: 보상 계산 및 Advantage 정규화
def calculate_rewards(responses: List[Dict], correct_answer: str) -> List[float]:
    """
    각 응답에 대한 보상 계산

    간단한 규칙: 정답이면 1.0, 오답이면 0.0
    실제로는 더 복잡한 보상 함수 사용 가능
    """
    rewards = []
    for resp in responses:
        if resp["answer"] == correct_answer:
            reward = 1.0
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards

def calculate_advantages(rewards: List[float]) -> List[float]:
    """
    GRPO의 그룹 상대적 Advantage 계산

    A_i = (r_i - μ) / σ
    """
    rewards = np.array(rewards)
    mean = np.mean(rewards)
    std = np.std(rewards)

    if std > 0:
        advantages = (rewards - mean) / std
    else:
        # 모든 보상이 같은 경우
        advantages = np.zeros_like(rewards)

    return advantages.tolist(), mean, std

# Step 2 실행
rewards = calculate_rewards(group, correct_answer="5")
advantages, mean_reward, std_reward = calculate_advantages(rewards)

print("📊 GRPO Step 2: 보상 및 Advantage 계산")
print("=" * 60)
print(f"그룹 통계: μ = {mean_reward:.2f}, σ = {std_reward:.2f}")
print("-" * 60)
print(f"{'응답':<8} {'정답여부':<10} {'보상':<8} {'Advantage':<12} {'액션':<15}")
print("-" * 60)

for i, (resp, reward, advantage) in enumerate(zip(group, rewards, advantages)):
    status = "✅ 정답" if resp["answer"] == "5" else "❌ 오답"
    if advantage > 0:
        action = "📈 확률 증가"
    elif advantage < 0:
        action = "📉 확률 감소"
    else:
        action = "➡️ 변화 없음"

    print(f"응답 {i+1:<3} {status:<10} {reward:<8.1f} {advantage:<+12.2f} {action}")
