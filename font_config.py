# 환경 설정: 한글 폰트 (Colab 환경)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ------------------------------------------------------------
# 🔧 한글 폰트 설정 (Colab용)
# ------------------------------------------------------------
# Colab에서는 아래 주석을 해제하여 폰트 다운로드 필요
# !wget 'https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf' -O 'NanumGothic.ttf' # 쥬피터노트북용
# 아래 범용 목적
import subprocess
subprocess.run([
    "wget",
    "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf",
    "-O", "NanumGothic.ttf"
])
# 아래 범용 목적 -END

# 폰트 파일이 있는 경우 등록
try:
    fm.fontManager.addfont("NanumGothic.ttf")
    plt.rc("font", family="NanumGothic")
except:
    print("⚠️ 한글 폰트 파일이 없습니다. Colab에서 wget 명령으로 다운로드하세요.")

# 마이너스 기호 깨짐 방지
plt.rc("axes", unicode_minus=False)
