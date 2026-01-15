# ============================================
# 📌 load_model.py
# 목적: 학습/추론에 사용할 모델과 토크나이저 로드
# - CONFIG["model_name"] 기반으로 AutoTokenizer, AutoModelForCausalLM 불러오기
# - 정밀도(fp16, bf16, fp32) 설정
# - 선택적으로 LoRA 적용 (CONFIG["use_lora"])
# ============================================

# 모델 로드
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch

print("🤖 모델 로드 중...")
print("=" * 50)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # 생성 모델에 권장

print(f"✅ 토크나이저 로드 완료: {CONFIG['model_name']}")

# 모델 로드 (정밀도 설정)
if CONFIG["precision"] == "bf16":
    dtype = torch.bfloat16
elif CONFIG["precision"] == "fp16":
    dtype = torch.float16
else:
    dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_name"],
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)

print(f"✅ 모델 로드 완료")
print(f"   파라미터 수: {model.num_parameters() / 1e6:.1f}M")

# LoRA 적용
if CONFIG["use_lora"]:
    lora_config = LoraConfig(
        r=16,                          # LoRA rank
        lora_alpha=32,                 # 스케일링 팩터
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n✅ LoRA 적용 완료")
    print(f"   학습 파라미터: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
