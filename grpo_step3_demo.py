# ============================================
# 📌 grpo_step3_demo.py
# 목적: GRPO Step 3 정책 업데이트 과정을 개념적으로 시뮬레이션
# - 실제 학습에서는 GRPOTrainer가 자동으로 처리
# - 교육용으로 클리핑과 KL 페널티의 역할을 설명하기 위해 작성
# 
# ============================================
# 📝 GRPO Step 3: 정책 업데이트 시뮬레이션
def simulate_policy_update(
    responses: List[Dict],
    advantages: List[float],
    epsilon: float = 0.2,
    beta: float = 0.01
) -> Dict:
    """
    GRPO 정책 업데이트 시뮬레이션

    실제로는 신경망 가중치가 업데이트됨
    여기서는 개념적 시뮬레이션
    """
    update_info = {
        "clipping_range": f"[{1-epsilon:.1f}, {1+epsilon:.1f}]",
        "kl_penalty_weight": beta,
        "updates": []
    }

    for i, (resp, adv) in enumerate(zip(responses, advantages)):
        # 시뮬레이션: 확률 비율 (실제로는 모델에서 계산)
        ratio = np.random.uniform(0.8, 1.2)

        # 클리핑
        clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)

        # 목적 함수 (둘 중 작은 값 선택)
        unclipped_obj = ratio * adv
        clipped_obj = clipped_ratio * adv
        objective = min(unclipped_obj, clipped_obj)

        update_info["updates"].append({
            "response_id": i + 1,
            "advantage": adv,
            "ratio": ratio,
            "clipped_ratio": clipped_ratio,
            "objective": objective,
            "was_clipped": abs(ratio - clipped_ratio) > 0.001
        })

    return update_info

# Step 3 실행
update_result = simulate_policy_update(group, advantages)

print("🔄 GRPO Step 3: 정책 업데이트")
print("=" * 60)
print(f"클리핑 범위: {update_result['clipping_range']}")
print(f"KL 페널티 가중치: {update_result['kl_penalty_weight']}")
print("-" * 60)

print(f"{'응답':<8} {'Advantage':<12} {'Ratio':<10} {'Clipped':<10} {'Objective':<12}")
print("-" * 60)

for update in update_result["updates"]:
    clip_mark = "📎" if update["was_clipped"] else ""
    print(f"응답 {update['response_id']:<3} {update['advantage']:<+12.2f} "
          f"{update['ratio']:<10.3f} {update['clipped_ratio']:<10.3f} "
          f"{update['objective']:<+12.3f} {clip_mark}")

print("\n💡 클리핑의 역할: ratio가 너무 크거나 작아지는 것을 방지하여 안정적 학습")
