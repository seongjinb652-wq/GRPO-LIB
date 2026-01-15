# ============================================
# 📌 grpo_pipeline_demo.py
# 목적: GRPO 학습 파이프라인 데모 코드
# - 모델 로드, LoRA 적용, 보상 함수 정의, Config 설정, Trainer 생성, 학습 실행, 저장까지 포함
# - 실제 프로젝트에서는 모듈화된 구조(run_grpo_pipeline.py) 사용 권장
# ============================================
# 📝 완전한 GRPO 학습 파이프라인 (데모용 - )

def create_grpo_pipeline_demo():
    """
    GRPO 학습 파이프라인 구성 데모

    이 함수는 파이프라인의 구조를 보여주기 위한 것으로,
    실제 학습은 다음 교시에서 진행합니다.
    """

    pipeline_code = '''
# === 1. 모델 및 토크나이저 로드 ===
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 기본 모델 로드 (작은 모델로 시작)
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # 메모리 효율
    device_map="auto",
)

# LoRA 적용 (메모리 효율적 파인튜닝)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# === 2. 보상 함수 정의 ===
def reward_function(completions, prompts, answer, **kwargs):
    """수학 문제 정답 + 형식 보상"""
    rewards = []
    for completion, ans in zip(completions, answer):
        # 정답 확인
        numbers = re.findall(r"-?\d+", completion)
        correct = 1.0 if numbers and numbers[-1] == ans else 0.0

        # 형식 확인
        has_think = 0.2 if "<think>" in completion else 0.0

        rewards.append(correct + has_think)
    return rewards

# === 3. GRPOConfig 설정 ===
from trl import GRPOConfig

training_args = GRPOConfig(
    output_dir="./grpo_math_model",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_generations=4,
    max_completion_length=256,
    beta=0.04,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    gradient_checkpointing=True,
)

# === 4. GRPOTrainer 생성 ===
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=reward_function,
)

# === 5. 학습 실행 ===
trainer.train()

# === 6. 모델 저장 ===
trainer.save_model("./grpo_math_model_final")
tokenizer.save_pretrained("./grpo_math_model_final")
'''

    return pipeline_code

# 파이프라인 코드 출력
print("🔄 GRPO 학습 파이프라인 구성")
print("=" * 60)
print(create_grpo_pipeline_demo())
print("=" * 60)

print("\n" + "=" * 60)
