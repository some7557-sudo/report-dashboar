"""
Groq → Gemini 전환 패치
C:\ReportDashboard\dashboard\ 에서 실행: python patch_gemini.py
"""
import shutil, os

APP_FILE = "app.py"
BACKUP = "app.py.groq_backup"

# 백업
if not os.path.exists(BACKUP):
    shutil.copy2(APP_FILE, BACKUP)
    print(f"✅ 백업 생성: {BACKUP}")

with open(APP_FILE, "r", encoding="utf-8") as f:
    code = f.read()

original = code  # 변경 감지용

# ── 1. 설정 변수 교체 ──
code = code.replace(
    'GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"',
    'GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"'
)
code = code.replace(
    'GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]',
    'GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite"]'
)

# ── 2. 모든 GROQ_ 참조를 GEMINI_로 교체 ──
code = code.replace("GROQ_API_KEY", "GEMINI_API_KEY")
code = code.replace("GROQ_API_URL", "GEMINI_API_URL")
code = code.replace("GROQ_MODELS", "GEMINI_MODELS")

# ── 3. 주석/로그 텍스트 교체 ──
code = code.replace("Groq 속도제한", "Gemini 속도제한")
code = code.replace("Groq로 요약", "Gemini로 요약")
code = code.replace("컨콜 텍스트를 Groq로 요약", "컨콜 텍스트를 Gemini로 요약")
code = code.replace("PDF → Groq", "PDF → Gemini")

# ── 결과 저장 ──
if code == original:
    print("⚠️ 변경 사항 없음 - 이미 패치됐거나 코드가 다릅니다")
else:
    with open(APP_FILE, "w", encoding="utf-8") as f:
        f.write(code)

    # 변경 통계
    changes = sum(1 for a, b in zip(original, code) if a != b)
    print(f"✅ 패치 완료!")
    print(f"   - Groq → Gemini 전환")
    print(f"   - 모델: gemini-2.0-flash (1순위), gemini-2.0-flash-lite (2순위)")
    print(f"   - 엔드포인트: Google Gemini OpenAI 호환 API")
    print(f"\n🔄 app.py 재시작 필요:")
    print(f"   1) 기존 app.py 종료 (작업관리자에서 python 종료)")
    print(f"   2) python app.py 다시 실행")
    print(f"\n💾 원복하려면: copy {BACKUP} {APP_FILE}")
