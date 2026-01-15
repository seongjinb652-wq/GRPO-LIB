# 학습된 모델로 추론 테스트
print("🧪 학습된 모델 추론 테스트")
print("=" * 60)

# 테스트 문제들
test_problems = [
    "다음을 계산하세요: 15 + 27 = ?",
    "다음을 계산하세요: 100 - 37 = ?",
    "다음을 계산하세요: 8 × 7 = ?",
    "철수가 사과 12개를 가지고 있었습니다. 영희에게 5개를 주었습니다. 남은 사과는?",
]

# 정답
correct_answers = ["42", "63", "56", "7"]

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """모델로 응답 생성"""
    full_prompt = f"""당신은 수학 문제를 단계별로 풀어주는 도우미입니다.
<think> 태그 안에 풀이 과정을 작성하고, 마지막에 "답: [숫자]" 형식으로 답을 제시하세요.

문제: {prompt}

"""
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 프롬프트 제거
    response = response[len(full_prompt):]
    return response.strip()

# 테스트 실행
print("테스트 결과:")
print("-" * 60)

correct_count = 0
for i, (problem, answer) in enumerate(zip(test_problems, correct_answers)):
    response = generate_response(model, tokenizer, problem)

    # 정답 확인
    numbers = re.findall(r'-?\d+', response)
    answer_match = re.search(r'답[:\s]*(-?\d+)', response)
    if answer_match:
        model_answer = answer_match.group(1)
    elif numbers:
        model_answer = numbers[-1]
    else:
        model_answer = "N/A"

    is_correct = model_answer == answer
    if is_correct:
        correct_count += 1

    status = "✅" if is_correct else "❌"

    print(f"\n[문제 {i+1}] {problem}")
    print(f"모델 응답: {response[:150]}...")
    print(f"정답: {answer} | 모델 답: {model_answer} {status}")

print("\n" + "=" * 60)
print(f"📊 정확도: {correct_count}/{len(test_problems)} ({100*correct_count/len(test_problems):.0f}%)")
