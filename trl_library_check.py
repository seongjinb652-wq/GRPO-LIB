# ============================================
# 📌 trl_info.py
# 목적: TRL 라이브러리 임포트 및 버전 확인
# - 현재 설치된 TRL 버전 출력
# - 제공되는 주요 Trainer 목록(SFT, DPO, PPO, GRPO) 확인
# ============================================
# TRL은 Hugging Face 생태계 덕분에 여전히 가장 많이 쓰이는 RLHF 라이브러리입니다. Hugging Face 생태계와 통합, 문서 풍부
# OpenRLHF는 최근 DeepSeek R1 같은 reasoning 모델에서 활용되며, TRL의 대안으로 급부상하고 있습니다. 대규모 분산 학습 지원, DeepSeek R1 등 사례
# LangChain은 RLHF 자체보다는 응용/워크플로우 관리에 많이 쓰여, 교육 과정에서 TRL과 함께 자주 등장합니다.파이프라인·에이전트 개발에 가장 많이 쓰임.  RLHF 직접 지원은 약하지만 응용에 필수
# Argilla/Mantis NLP는 데이터 라벨링과 피드백 관리에 강점이 있어 RLHF 실습 과정에서 보조적으로 활용됩니다.
# ============================================
# TRL 라이브러리 임포트 및 버전 확인
import trl
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

print("📚 TRL 라이브러리 정보")
print("=" * 50)
print(f"TRL 버전: {trl.__version__}")

# 사용 가능한 Trainer 목록
trainers = [
    ("SFTTrainer", "지도학습 파인튜닝"),
    ("DPOTrainer", "직접 선호도 최적화"),
    ("PPOTrainer", "근접 정책 최적화"),
    ("GRPOTrainer", "그룹 상대적 정책 최적화"),
]

print("\n📋 TRL에서 제공하는 주요 Trainer:")
for name, desc in trainers:
    print(f"  • {name}: {desc}")
