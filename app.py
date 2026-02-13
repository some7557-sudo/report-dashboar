"""
?“Š ì¦ê¶Œ??ë¦¬í¬???€?œë³´??v3
- 3ê°€ì§€ ? í˜•: ì¢…ëª©/?„ëµ/?°ì—…
- ?¬ì?˜ê²¬ ë³€ê²??˜ì´?¼ì´??- PDF ?…ë¡œ??+ ?”ë ˆê·¸ë¨ ë´??°ë™
"""

import os, sys, json, time, sqlite3, threading, logging, re, secrets
from datetime import datetime, timedelta
from pathlib import Path

try:
    from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
except ImportError:
    print("??pip install flask"); sys.exit(1)
try:
    import httpx
except ImportError:
    print("??pip install httpx"); sys.exit(1)
try:
    import fitz
except ImportError:
    print("??pip install PyMuPDF"); sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from pykrx import stock as krx_stock
    HAS_PYKRX = True
except ImportError:
    HAS_PYKRX = False

# ?€?€ ?¤ì • ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / "reports.db"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.secret_key = os.getenv("SECRET_KEY", "jason-report-dashboard-2026-secret")

# ?€?€ Google OAuth ?¤ì • ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "145923848376-7l7tpmsi19qj5l71nvsidbrjnsp4kfu6.apps.googleusercontent.com")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "https://jason-report.mooo.com/callback")
ALLOWED_EMAILS = [e.strip().lower() for e in os.getenv("ALLOWED_EMAILS", "hanju723@gmail.com,lhjkd13@gmail.com,firerjatls@gmail.com,selba2240@gmail.com,some7557@gmail.com").split(",")]

from functools import wraps
from flask import session

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_email"):
            return redirect(url_for("login_page", next=request.url))
        return f(*args, **kwargs)
    return decorated

# PDF ?Œì¼ëª?URL ?¸ì½”???„í„°
from urllib.parse import quote as url_quote
app.jinja_env.filters["urlquote"] = lambda s: url_quote(str(s), safe="")

last_api_call = 0
MIN_INTERVAL = 3


def make_safe_filename(original_name):
    """?œê? ?Œì¼ëª…ì„ ASCII-safe ?´ë¦„?¼ë¡œ ë³€??""
    import hashlib
    ext = Path(original_name).suffix or ".pdf"
    ts = int(time.time())
    h = hashlib.md5(original_name.encode()).hexdigest()[:8]
    return f"{ts}_{h}{ext}"


# ?€?€ DB ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type TEXT DEFAULT 'stock',
            report_title TEXT,
            stock_name TEXT,
            ticker TEXT,
            brokerage TEXT,
            analyst TEXT,
            rating TEXT,
            prev_rating TEXT,
            target_price TEXT,
            prev_target_price TEXT,
            current_price TEXT,
            report_date TEXT,
            category TEXT DEFAULT 'KR',
            sector TEXT,
            headline TEXT,
            key_points TEXT,
            mentioned_stocks TEXT,
            risks TEXT,
            summary TEXT,
            earnings_estimates TEXT,
            pdf_filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    # ê¸°ì¡´ DB??ì»¬ëŸ¼ ì¶”ê?
    try:
        conn.execute("ALTER TABLE reports ADD COLUMN earnings_estimates TEXT DEFAULT ''")
        conn.commit()
    except:
        pass
    try:
        conn.execute("ALTER TABLE reports ADD COLUMN source_url TEXT DEFAULT ''")
        conn.commit()
    except:
        pass

    # ì»¨ì½œ ?Œì´ë¸?    conn.execute("""
        CREATE TABLE IF NOT EXISTS concalls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            host TEXT,
            concall_time TEXT,
            concall_date TEXT,
            clova_url TEXT,
            transcript TEXT,
            headline TEXT,
            key_points TEXT,
            guidance TEXT,
            guidance_numbers TEXT,
            earnings_surprise TEXT,
            qa_highlights TEXT,
            keywords TEXT,
            sentiment TEXT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    # ê¸°ì¡´ concalls DB ì»¬ëŸ¼ ì¶”ê?
    for col in ["guidance_numbers", "earnings_surprise", "keywords"]:
        try:
            conn.execute(f"ALTER TABLE concalls ADD COLUMN {col} TEXT DEFAULT ''")
            conn.commit()
        except:
            pass

    # ì£¼ê? ìºì‹œ ?Œì´ë¸?    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            report_date TEXT,
            price_on_date REAL,
            price_d1 REAL,
            price_d3 REAL,
            price_d5 REAL,
            change_d1 REAL,
            change_d3 REAL,
            change_d5 REAL,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, report_date)
        )
    """)
    conn.commit()
    conn.close()


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ?€?€ ì£¼ê? ?°ì´???€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def fetch_stock_price(ticker, report_date_str):
    """ë¦¬í¬??ë°œí–‰???„í›„ ì£¼ê? ë³€??ì¡°íšŒ (pykrx)"""
    if not HAS_PYKRX or not ticker:
        return None
    code = ticker.replace(".KS", "").replace(".KQ", "").strip()
    if not code or not code.isdigit() or len(code) != 6:
        return None

    # ìºì‹œ ?•ì¸
    conn = get_db()
    cached = conn.execute("SELECT * FROM price_cache WHERE ticker = ? AND report_date = ?",
                          [code, report_date_str]).fetchone()
    if cached:
        conn.close()
        return dict(cached)
    conn.close()

    try:
        rd = datetime.strptime(report_date_str, "%Y-%m-%d")
        start = (rd - timedelta(days=7)).strftime("%Y%m%d")
        end = (rd + timedelta(days=12)).strftime("%Y%m%d")
        df = krx_stock.get_market_ohlcv(start, end, code)
        if df.empty or len(df) < 2:
            return None

        # ë¦¬í¬??ë°œí–‰???ëŠ” ì§ì „ ê±°ë˜??ì°¾ê¸°
        rd_fmt = rd.strftime("%Y-%m-%d")
        dates = [d.strftime("%Y-%m-%d") for d in df.index]

        base_idx = None
        for i, d in enumerate(dates):
            if d >= rd_fmt:
                base_idx = i
                break
        if base_idx is None:
            base_idx = len(dates) - 1

        base_price = float(df.iloc[base_idx]["ì¢…ê?"])
        result = {
            "ticker": code, "report_date": report_date_str,
            "price_on_date": base_price,
            "price_d1": None, "price_d3": None, "price_d5": None,
            "change_d1": None, "change_d3": None, "change_d5": None,
        }

        for offset, key in [(1, "d1"), (3, "d3"), (5, "d5")]:
            idx = base_idx + offset
            if idx < len(df):
                p = float(df.iloc[idx]["ì¢…ê?"])
                result[f"price_{key}"] = p
                result[f"change_{key}"] = round((p - base_price) / base_price * 100, 2)

        # ìºì‹œ ?€??        conn = get_db()
        try:
            conn.execute("""INSERT OR REPLACE INTO price_cache
                (ticker,report_date,price_on_date,price_d1,price_d3,price_d5,change_d1,change_d3,change_d5)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                [code, report_date_str, result["price_on_date"],
                 result["price_d1"], result["price_d3"], result["price_d5"],
                 result["change_d1"], result["change_d3"], result["change_d5"]])
            conn.commit()
        except:
            pass
        conn.close()
        return result
    except Exception as e:
        logger.debug(f"Price fetch fail {code}: {e}")
        return None


def get_prices_batch(reports):
    """ë¦¬í¬??ë¦¬ìŠ¤?¸ì— ì£¼ê? ?°ì´???¼ê´„ ì²¨ë?"""
    if not HAS_PYKRX:
        return reports
    for r in reports:
        if r.get("report_type") == "stock" and r.get("ticker"):
            r["price_data"] = fetch_stock_price(r["ticker"], r.get("report_date", ""))
        else:
            r["price_data"] = None
    return reports


# ?€?€ PDF ??Gemini ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def extract_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for i in range(min(len(doc), 10)):
        parts.append(doc[i].get_text())
    doc.close()
    text = "\n".join(parts)
    return text[:10000] if len(text) > 10000 else text


def wait_rate():
    global last_api_call
    elapsed = time.time() - last_api_call
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    last_api_call = time.time()


def parse_pdf(pdf_bytes):
    pdf_text = extract_text(pdf_bytes)
    if not pdf_text.strip():
        raise Exception("PDF ?ìŠ¤??ì¶”ì¶œ ?¤íŒ¨")

    prompt = f"""You are a Korean equity research analyst. Parse this brokerage report.

RULES:
- ALL output MUST be in Korean. Translate English.
- Include specific numbers/percentages.
- Carefully determine the report type.
- For stock reports: extract 2026E/2027E/2028E earnings estimates from tables. Use ì¡?for trillion, ??for hundred million. Include units. If a year is missing, use empty strings.

Return ONLY valid JSON:

{{
  "report_type": "stock/industry/strategy - stock=ê°œë³„ì¢…ëª©, industry=?°ì—…ë¶„ì„(?¬ëŸ¬ì¢…ëª©ì»¤ë²„), strategy=?œì¥?„ëµ/ë§¤í¬ë¡?,
  "report_title": "?œêµ­???œëª©",
  "stock_name": "ì£??€??ì¢…ëª©ëª??œêµ­??. industryë©?'?œêµ­ ë©”ëª¨ë¦??ŒìŠ¤???¥ë¹„' ?? strategyë©?'MSCI Korea' ??,
  "ticker": "ì¢…ëª©ì½”ë“œ(?ˆìœ¼ë©?. ?? 090430.KS",
  "brokerage": "ì¦ê¶Œ?¬ëª…(?ë¬¸? ì?)",
  "analyst": "? ë„ë¦¬ìŠ¤???¬ëŸ¬ëª…ì´ë©??¼í‘œ)",
  "rating": "?„ì¬ ?¬ì?˜ê²¬. Buy/Overweight/Neutral/Hold/Sell/Underweight/Outperform",
  "prev_rating": "?´ì „ ?¬ì?˜ê²¬(ë³€ê²½ì‹œ). ?†ìœ¼ë©?ë¹?ë¬¸ì?? ?? Neutral",
  "target_price": "ëª©í‘œê°€(?µí™”?¬í•¨). ?? 180,000??,
  "prev_target_price": "?´ì „ ëª©í‘œê°€(ë³€ê²½ì‹œ). ?†ìœ¼ë©?ë¹?ë¬¸ì??,
  "current_price": "?„ì¬ê°€. ?†ìœ¼ë©?ë¹?ë¬¸ì??,
  "report_date": "YYYY-MM-DD",
  "category": "KR/GL/ETC",
  "sector": "?…ì¢… ?œêµ­?? ?? ë°˜ë„ì²? ?”ì¥??ë·°í‹°, ?„ë ¥ê¸°ê¸°, ?œì¥?„ëµ",
  "headline": "?µì‹¬ ??ì¤?(40??. ê°€??ì¤‘ìš”??ê²°ë¡ ",
  "key_points": ["?µì‹¬ 3~5ê°? êµ¬ì²´???˜ì¹˜ ?„ìˆ˜. ì§§ê³  ?„íŒ©?¸ìˆê²?],
  "mentioned_stocks": "?¸ê¸‰??ì¢…ëª©+?˜ê²¬ (industry/strategy??. ?? 'DI Corp Buy TP 45,000??/ ISC Buy TP 50,000??. stock?´ë©´ ë¹?ë¬¸ì??,
  "risks": "ë¦¬ìŠ¤??1~2ì¤?,
  "earnings_estimates": {{
    "2026E": {{"revenue": "ë§¤ì¶œ??ì¡??µì›)", "op": "?ì—…?´ìµ(ì¡??µì›)", "np": "?œì´??ì¡??µì›)", "eps": "EPS(??", "per": "PER(ë°?"}},
    "2027E": {{"revenue": "", "op": "", "np": "", "eps": "", "per": ""}},
    "2028E": {{"revenue": "", "op": "", "np": "", "eps": "", "per": ""}}
  }},
  "summary": "2~3ë¬¸ì¥ ?µì‹¬ ?”ì•½. ?¬ì?ê? 30ì´??ˆì— ?Œì•… ê°€?¥í•˜?„ë¡"
}}

ONLY JSON. No markdown.

--- REPORT ---
{pdf_text}"""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    for model in GROQ_MODELS:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1, "max_tokens": 2000}
        try:
            wait_rate()
            logger.info(f"Trying: {model}")
            with httpx.Client(timeout=60) as client:
                resp = client.post(GROQ_API_URL, headers=headers, json=body)
            if resp.status_code in (404, 413):
                continue
            if resp.status_code == 429:
                time.sleep(30); continue
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            logger.info(f"??Parsed with {model}")
            return json.loads(text)
        except json.JSONDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error {model}: {e}")
            continue

    raise Exception("ëª¨ë“  ëª¨ë¸ ?œë„ ?¤íŒ¨")


def notify_all_chats(text):
    """ëª¨ë“  ?±ë¡??chat_id???Œë¦¼ ë°œì†¡"""
    if not TELEGRAM_TOKEN:
        return
    cid_file = BASE_DIR / "chat_ids.txt"
    if not cid_file.exists():
        return
    chat_ids = [l.strip() for l in cid_file.read_text().splitlines() if l.strip()]
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for cid in chat_ids:
        try:
            httpx.post(url, json={"chat_id": cid, "text": text,
                      "parse_mode": "Markdown", "disable_web_page_preview": True}, timeout=10)
        except:
            pass
    logger.info(f"?“¢ ?Œë¦¼ ë°œì†¡: {len(chat_ids)}ëª?)


def save_report(data, pdf_filename=""):
    conn = get_db()

    # ì¤‘ë³µ ì²´í¬: ê°™ì? ì¢…ëª© + ì¦ê¶Œ??+ ? ì§œ + ?œëª©
    stock = data.get("stock_name", "").strip()
    broker = data.get("brokerage", "").strip()
    rdate = data.get("report_date", "").strip()
    title = data.get("report_title", "").strip()

    if stock and broker and rdate:
        existing = conn.execute(
            """SELECT id FROM reports
               WHERE stock_name = ? AND brokerage = ? AND report_date = ? AND report_title = ?""",
            [stock, broker, rdate, title]
        ).fetchone()
        if existing:
            logger.info(f"? ï¸ ì¤‘ë³µ ë¦¬í¬???¤í‚µ: {stock} - {broker} - {rdate}")
            conn.close()
            return None

    kp = json.dumps(data.get("key_points", []), ensure_ascii=False)
    conn.execute("""
        INSERT INTO reports (report_type, report_title, stock_name, ticker, brokerage, analyst,
            rating, prev_rating, target_price, prev_target_price, current_price,
            report_date, category, sector, headline, key_points, mentioned_stocks,
            risks, summary, earnings_estimates, pdf_filename)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data.get("report_type", "stock"),
        data.get("report_title", ""),
        data.get("stock_name", ""),
        data.get("ticker", ""),
        data.get("brokerage", ""),
        data.get("analyst", ""),
        data.get("rating", ""),
        data.get("prev_rating", ""),
        data.get("target_price", ""),
        data.get("prev_target_price", ""),
        data.get("current_price", ""),
        data.get("report_date", ""),
        data.get("category", "KR"),
        data.get("sector", ""),
        data.get("headline", ""),
        kp,
        data.get("mentioned_stocks", ""),
        data.get("risks", ""),
        data.get("summary", ""),
        json.dumps(data.get("earnings_estimates", {}), ensure_ascii=False),
        pdf_filename,
    ))
    conn.commit()
    rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    # ?¤ì‹œê°??Œë¦¼
    try:
        emoji = {"Buy": "?Ÿ¢", "Overweight": "?Ÿ¢", "Outperform": "?Ÿ¢",
                "Neutral": "?Ÿ¡", "Hold": "?Ÿ¡", "Sell": "?”´", "Underweight": "?”´"}.get(data.get("rating", ""), "??)
        tp = f" TP {data['target_price']}" if data.get("target_price") else ""
        msg = f"?“„ *??ë¦¬í¬???±ë¡*\n"
        msg += f"{emoji} *{data.get('stock_name', '')}* {data.get('brokerage', '')} {data.get('rating', '')}{tp}\n"
        if data.get("headline"):
            msg += f"?’¡ {data['headline']}\n"
        dash_url = get_dashboard_url()
        msg += f"?”— [?€?œë³´??({dash_url}/dashboard)"
        notify_all_chats(msg)
    except Exception as e:
        logger.error(f"ë¦¬í¬???Œë¦¼ ?¤íŒ¨: {e}")

    return rid
    if not NOTION_TOKEN or not NOTION_DATABASE_ID:
        return
    try:
        kp_str = ""
        if isinstance(data.get("key_points"), list):
            kp_str = "\n".join(f"??{p}" for p in data["key_points"])
        properties = {
            "Report Title": {"title": [{"text": {"content": data.get("report_title", "Untitled")[:2000]}}]},
            "Stock Name": {"rich_text": [{"text": {"content": data.get("stock_name", "")[:2000]}}]},
            "Ticker": {"rich_text": [{"text": {"content": data.get("ticker", "")[:2000]}}]},
            "Analyst": {"rich_text": [{"text": {"content": data.get("analyst", "")[:2000]}}]},
            "Target Price": {"rich_text": [{"text": {"content": data.get("target_price", "")[:2000]}}]},
            "Key Points": {"rich_text": [{"text": {"content": kp_str[:2000]}}]},
        }
        if data.get("brokerage"):
            properties["Brokerage"] = {"select": {"name": data["brokerage"]}}
        if data.get("rating"):
            properties["Rating"] = {"select": {"name": data["rating"]}}
        if data.get("category"):
            properties["Category"] = {"select": {"name": data["category"]}}
        if data.get("report_date"):
            properties["Report Date"] = {"date": {"start": data["report_date"]}}
        body = {"parent": {"database_id": NOTION_DATABASE_ID}, "properties": properties}
        headers = {"Authorization": f"Bearer {NOTION_TOKEN}", "Content-Type": "application/json",
                   "Notion-Version": "2022-06-28"}
        with httpx.Client(timeout=30) as client:
            client.post("https://api.notion.com/v1/pages", headers=headers, json=body)
        logger.info("??Notion ?€??)
    except Exception as e:
        logger.warning(f"Notion ?¤íŒ¨: {e}")


# ?€?€ Flask ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
@app.route("/login", methods=["GET"])
def login_page():
    next_url = request.args.get("next", "/")
    # Google OAuth URL ?ì„±
    state = secrets.token_urlsafe(32)
    session["oauth_state"] = state
    session["oauth_next"] = next_url
    google_auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        "&response_type=code"
        "&scope=email profile"
        f"&state={state}"
        "&access_type=offline"
        "&prompt=select_account"
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ë¡œê·¸??/title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Noto Sans KR',sans-serif;background:#0a0e17;color:#e0e6f0;
    display:flex;justify-content:center;align-items:center;min-height:100vh}}
.box{{background:#0f1520;border:1px solid #1e2a42;border-radius:16px;padding:40px;width:360px;text-align:center}}
h1{{font-size:20px;font-weight:900;margin-bottom:6px}}
.sub{{color:#4a5a7a;font-size:13px;margin-bottom:28px}}
.google-btn{{display:flex;align-items:center;justify-content:center;gap:10px;width:100%;
    background:#fff;color:#333;border:none;padding:12px;border-radius:10px;
    font-size:14px;font-weight:600;cursor:pointer;font-family:inherit;text-decoration:none}}
.google-btn:hover{{background:#f0f0f0}}
.google-btn img{{width:20px;height:20px}}
</style></head><body>
<div class="box">
    <h1>?“Š ë¦¬í¬???€?œë³´??/h1>
    <div class="sub">Google ê³„ì •?¼ë¡œ ë¡œê·¸?¸í•˜?¸ìš”</div>
    <a href="{google_auth_url}" class="google-btn">
        <img src="https://www.google.com/favicon.ico" alt="G">
        Googleë¡?ë¡œê·¸??    </a>
</div>
</body></html>"""


@app.route("/callback")
def google_callback():
    code = request.args.get("code")
    state = request.args.get("state")
    error = request.args.get("error")

    if error:
        return redirect("/login")

    # state ê²€ì¦?    if state != session.get("oauth_state"):
        return redirect("/login")

    # code ??token êµí™˜
    try:
        token_resp = httpx.post("https://oauth2.googleapis.com/token", data={
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        }, timeout=10)
        token_data = token_resp.json()
        access_token = token_data.get("access_token")

        if not access_token:
            logger.error(f"OAuth token error: {token_data}")
            return redirect("/login")

        # ?¬ìš©???•ë³´ ê°€?¸ì˜¤ê¸?        user_resp = httpx.get("https://www.googleapis.com/oauth2/v2/userinfo",
                              headers={"Authorization": f"Bearer {access_token}"}, timeout=10)
        user_data = user_resp.json()
        email = user_data.get("email", "").lower()
        name = user_data.get("name", "")

        logger.info(f"?” Google ë¡œê·¸???œë„: {email} ({name})")

        # ?´ë©”???ˆìš© ?•ì¸
        if email not in ALLOWED_EMAILS:
            logger.warning(f"??ë¹„í—ˆ???´ë©”?? {email}")
            return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>?‘ê·¼ ê±°ë?</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Noto Sans KR',sans-serif;background:#0a0e17;color:#e0e6f0;
    display:flex;justify-content:center;align-items:center;min-height:100vh}}
.box{{background:#0f1520;border:1px solid #1e2a42;border-radius:16px;padding:40px;width:360px;text-align:center}}
h2{{color:#f87171;margin-bottom:12px}}
p{{color:#4a5a7a;font-size:13px;margin-bottom:20px}}
a{{color:#3b82f6;text-decoration:none}}
</style></head><body>
<div class="box">
    <h2>???‘ê·¼ ê¶Œí•œ ?†ìŒ</h2>
    <p>{email} ê³„ì •?€<br>?‘ê·¼???ˆìš©?˜ì? ?Šì•˜?µë‹ˆ??</p>
    <a href="/login">?¤ë¥¸ ê³„ì •?¼ë¡œ ë¡œê·¸??/a>
</div>
</body></html>""", 403

        # ?¸ì…˜ ?¤ì •
        session["user_email"] = email
        session["user_name"] = name
        session.permanent = True
        app.permanent_session_lifetime = timedelta(days=30)

        next_url = session.pop("oauth_next", "/")
        session.pop("oauth_state", None)
        logger.info(f"??ë¡œê·¸???±ê³µ: {email}")
        return redirect(next_url)

    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return redirect("/login")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/")
@login_required
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
@login_required
def dashboard():
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    category = request.args.get("category", "")
    search = request.args.get("search", "")
    stock = request.args.get("stock", "")
    rtype = request.args.get("type", "")

    conn = get_db()

    query = "SELECT * FROM reports WHERE report_date = ?"
    params = [date_str]

    if category:
        query += " AND category = ?"
        params.append(category)
    if rtype:
        query += " AND report_type = ?"
        params.append(rtype)
    if search:
        query += " AND (stock_name LIKE ? OR analyst LIKE ? OR brokerage LIKE ? OR report_title LIKE ? OR headline LIKE ?)"
        params.extend([f"%{search}%"] * 5)
    if stock:
        query += " AND stock_name = ?"
        params.append(stock)

    query += " ORDER BY report_type ASC, created_at DESC"
    reports = conn.execute(query, params).fetchall()

    # ì¹´í…Œê³ ë¦¬ ì¹´ìš´??    counts = {}
    for cat in ["KR", "GL", "ETC"]:
        r = conn.execute("SELECT COUNT(*) FROM reports WHERE report_date = ? AND category = ?", [date_str, cat]).fetchone()
        counts[cat] = r[0]

    # ? í˜•ë³?ì¹´ìš´??    type_counts = {}
    for t in ["stock", "industry", "strategy"]:
        r = conn.execute("SELECT COUNT(*) FROM reports WHERE report_date = ? AND report_type = ?", [date_str, t]).fetchone()
        type_counts[t] = r[0]

    # ì¢…ëª©ë³?ì¹´ìš´??    stock_counts = conn.execute("""
        SELECT stock_name, COUNT(*) as cnt FROM reports
        WHERE report_date = ? AND stock_name != '' AND report_type = 'stock'
        GROUP BY stock_name ORDER BY cnt DESC LIMIT 15
    """, [date_str]).fetchall()

    # ?¬ì?˜ê²¬ ë³€ê²?ê±´ìˆ˜
    upgrade_count = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE report_date = ? AND prev_rating != ''", [date_str]
    ).fetchone()[0]

    total = conn.execute("SELECT COUNT(*) FROM reports WHERE report_date = ?", [date_str]).fetchone()[0]

    # ë¦¬í¬??ê·¸ë£¹??    report_list = []
    for r in reports:
        d = dict(r)
        try:
            d["key_points_list"] = json.loads(d.get("key_points", "[]"))
        except:
            d["key_points_list"] = []
        d["has_rating_change"] = bool(d.get("prev_rating"))
        d["has_tp_change"] = bool(d.get("prev_target_price"))
        try:
            d["earnings"] = json.loads(d.get("earnings_estimates", "{}") or "{}")
        except:
            d["earnings"] = {}
        report_list.append(d)

    conn.close()

    # ì£¼ê? ?°ì´??ì²¨ë? (ë°±ê·¸?¼ìš´?œì—??ë¹„ë™ê¸?ì²˜ë¦¬?˜ì? ?Šê³  ?™ê¸° - ìºì‹œ ?œìš©)
    report_list = get_prices_batch(report_list)

    try:
        current_date = datetime.strptime(date_str, "%Y-%m-%d")
    except:
        current_date = datetime.now()

    prev_date = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
    next_date = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    weekdays_kr = ["??, "??, "??, "ëª?, "ê¸?, "??, "??]
    weekday = weekdays_kr[current_date.weekday()]

    # ? í˜•ë³?ë¶„ë¥˜
    strategy_reports = [r for r in report_list if r["report_type"] == "strategy"]
    industry_reports = [r for r in report_list if r["report_type"] == "industry"]
    stock_reports = [r for r in report_list if r["report_type"] == "stock"]
    rating_changed = [r for r in report_list if r["has_rating_change"]]

    return render_template("dashboard.html",
        reports=report_list,
        strategy_reports=strategy_reports,
        industry_reports=industry_reports,
        stock_reports=stock_reports,
        rating_changed=rating_changed,
        date_str=date_str, prev_date=prev_date, next_date=next_date,
        today=today, weekday=weekday, current_date=current_date,
        counts=counts, type_counts=type_counts,
        stock_counts=stock_counts, upgrade_count=upgrade_count,
        category=category, search=search, stock_filter=stock, rtype=rtype,
        total=total,
    )


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "GET":
        return render_template("upload.html")

    files = request.files.getlist("pdfs")
    if not files:
        return jsonify({"error": "?Œì¼ ?†ìŒ"}), 400

    results = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            results.append({"file": f.filename, "status": "error", "msg": "PDFê°€ ?„ë‹™?ˆë‹¤"})
            continue
        try:
            pdf_bytes = f.read()
            safe_name = make_safe_filename(f.filename)
            (UPLOAD_DIR / safe_name).write_bytes(pdf_bytes)
            parsed = parse_pdf(pdf_bytes)
            rid = save_report(parsed, safe_name)
            if rid is None:
                # ì¤‘ë³µ ??PDF ?Œì¼ ?? œ
                (UPLOAD_DIR / safe_name).unlink(missing_ok=True)
                results.append({"file": f.filename, "status": "duplicate",
                              "stock": parsed.get("stock_name", ""), "brokerage": parsed.get("brokerage", "")})
            else:
                save_to_notion(parsed)
                results.append({"file": f.filename, "status": "success",
                              "stock": parsed.get("stock_name", ""), "brokerage": parsed.get("brokerage", ""), "id": rid})
        except Exception as e:
            logger.error(f"Error {f.filename}: {e}")
            results.append({"file": f.filename, "status": "error", "msg": str(e)[:200]})

    if request.headers.get("Accept") == "application/json":
        return jsonify(results)
    return render_template("upload.html", results=results)


@app.route("/pdf/<path:filename>")
@login_required
def serve_pdf(filename):
    from urllib.parse import unquote
    decoded = unquote(filename)
    # ?Œì¼ ì¡´ì¬ ?•ì¸
    fpath = UPLOAD_DIR / decoded
    if fpath.exists():
        return send_from_directory(str(UPLOAD_DIR), decoded)
    # URL?¸ì½”?©ëœ ?´ë¦„?¼ë¡œ???œë„
    fpath2 = UPLOAD_DIR / filename
    if fpath2.exists():
        return send_from_directory(str(UPLOAD_DIR), filename)
    # ?€?„ìŠ¤?¬í”„ prefixë¡?ê²€??(ë¶€ë¶?ë§¤ì¹­)
    prefix = filename.split("_")[0] if "_" in filename else ""
    if prefix and prefix.isdigit():
        for f in UPLOAD_DIR.iterdir():
            if f.name.startswith(prefix):
                return send_from_directory(str(UPLOAD_DIR), f.name)
    return "PDF not found", 404


@app.route("/api/reports")
@login_required
def api_reports():
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    conn = get_db()
    rows = conn.execute("SELECT * FROM reports WHERE report_date = ? ORDER BY created_at DESC", [date_str]).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/search")
@login_required
def api_search():
    q = request.args.get("q", "").strip()
    if not q or len(q) < 1:
        return jsonify([])

    conn = get_db()
    results = []

    # ë¦¬í¬??ê²€??    try:
        rows = conn.execute(
            """SELECT id, stock_name, report_title, brokerage, report_date, report_type, pdf_filename
               FROM reports
               WHERE stock_name LIKE ? OR report_title LIKE ? OR brokerage LIKE ?
               ORDER BY report_date DESC LIMIT 20""",
            [f"%{q}%", f"%{q}%", f"%{q}%"]
        ).fetchall()
        for r in rows:
            results.append({
                "type": "report",
                "id": r["id"],
                "name": r["stock_name"] or r["report_title"] or "",
                "sub": f"{r['brokerage'] or ''} Â· {r['report_date'] or ''}",
                "date": r["report_date"] or "",
                "url": f"/dashboard?date={r['report_date']}#report-{r['id']}",
                "headline": r["report_title"] or "",
            })
    except Exception as e:
        logger.error(f"ë¦¬í¬??ê²€???ëŸ¬: {e}")

    # ì»¨ì½œ ê²€??    try:
        crows = conn.execute(
            """SELECT id, company_name, host, concall_date, headline
               FROM concalls
               WHERE company_name LIKE ? OR host LIKE ? OR headline LIKE ?
               ORDER BY concall_date DESC LIMIT 20""",
            [f"%{q}%", f"%{q}%", f"%{q}%"]
        ).fetchall()
        for c in crows:
            results.append({
                "type": "concall",
                "id": c["id"],
                "name": c["company_name"] or "ì»¨ì½œ",
                "sub": f"{c['host'] or ''} Â· {c['concall_date'] or ''}",
                "date": c["concall_date"] or "",
                "url": f"/concalls?date={c['concall_date']}#concall-{c['id']}",
                "headline": c["headline"] or "",
            })
    except Exception as e:
        logger.error(f"ì»¨ì½œ ê²€???ëŸ¬: {e}")

    conn.close()
    results.sort(key=lambda x: x.get("date", "") or "", reverse=True)
    return jsonify(results[:30])


@app.route("/delete/<int:report_id>", methods=["POST"])
@login_required
def delete_report(report_id):
    conn = get_db()
    row = conn.execute("SELECT pdf_filename, report_date FROM reports WHERE id = ?", [report_id]).fetchone()
    if row:
        # PDF ?Œì¼???? œ
        if row["pdf_filename"]:
            pdf_path = UPLOAD_DIR / row["pdf_filename"]
            if pdf_path.exists():
                pdf_path.unlink()
        conn.execute("DELETE FROM reports WHERE id = ?", [report_id])
        conn.commit()
        date = row["report_date"] or datetime.now().strftime("%Y-%m-%d")
    else:
        date = datetime.now().strftime("%Y-%m-%d")
    conn.close()
    return redirect(f"/dashboard?date={date}")


@app.route("/delete-all", methods=["POST"])
@login_required
def delete_all():
    date_str = request.form.get("date", "")
    conn = get_db()
    if date_str:
        rows = conn.execute("SELECT pdf_filename FROM reports WHERE report_date = ?", [date_str]).fetchall()
        conn.execute("DELETE FROM reports WHERE report_date = ?", [date_str])
    else:
        rows = conn.execute("SELECT pdf_filename FROM reports").fetchall()
        conn.execute("DELETE FROM reports")
    conn.commit()
    conn.close()
    # PDF ?Œì¼ ?? œ
    for r in rows:
        if r["pdf_filename"]:
            p = UPLOAD_DIR / r["pdf_filename"]
            if p.exists():
                p.unlink()
    return redirect(f"/dashboard?date={date_str}" if date_str else "/dashboard")


@app.route("/cleanup-duplicates", methods=["GET", "POST"])
@login_required
def cleanup_duplicates():
    """ê¸°ì¡´ ì¤‘ë³µ ë¦¬í¬???•ë¦¬ - ê°™ì? ì¢…ëª©+ì¦ê¶Œ??? ì§œ+?œëª©?´ë©´ ìµœì‹  1ê°œë§Œ ?¨ê?"""
    conn = get_db()
    dupes = conn.execute("""
        SELECT stock_name, brokerage, report_date, report_title, COUNT(*) as cnt, GROUP_CONCAT(id) as ids
        FROM reports
        WHERE stock_name != ''
        GROUP BY stock_name, brokerage, report_date, report_title
        HAVING cnt > 1
    """).fetchall()
    deleted = 0
    for d in dupes:
        ids = sorted([int(x) for x in d["ids"].split(",")])
        keep_id = ids[-1]  # ê°€??ìµœê·¼ ID ? ì?
        for del_id in ids[:-1]:
            row = conn.execute("SELECT pdf_filename FROM reports WHERE id = ?", [del_id]).fetchone()
            if row and row["pdf_filename"]:
                p = UPLOAD_DIR / row["pdf_filename"]
                if p.exists():
                    p.unlink()
            conn.execute("DELETE FROM reports WHERE id = ?", [del_id])
            deleted += 1
    conn.commit()
    conn.close()
    logger.info(f"?§¹ ì¤‘ë³µ ?•ë¦¬: {deleted}ê±??? œ")
    return redirect(f"/dashboard?msg=ì¤‘ë³µ {deleted}ê±??•ë¦¬ ?„ë£Œ")


# ?€?€ ì¢…ëª© ë¹„êµ ?˜ì´ì§€ ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
@app.route("/compare")
@login_required
def compare():
    stock = request.args.get("stock", "")
    conn = get_db()
    # ë¹„êµ??ì¢…ëª© ëª©ë¡
    stocks = conn.execute("""
        SELECT stock_name, COUNT(*) as cnt, COUNT(DISTINCT brokerage) as brokers
        FROM reports WHERE stock_name != '' AND report_type = 'stock'
        GROUP BY stock_name HAVING cnt > 1 ORDER BY cnt DESC LIMIT 30
    """).fetchall()

    rows = []
    if stock:
        rows = conn.execute("""
            SELECT * FROM reports WHERE stock_name = ? AND report_type = 'stock'
            ORDER BY report_date DESC, brokerage
        """, [stock]).fetchall()
    conn.close()

    reports = []
    for r in rows:
        d = dict(r)
        try:
            d["key_points_list"] = json.loads(d.get("key_points", "[]"))
        except:
            d["key_points_list"] = []
        try:
            d["earnings"] = json.loads(d.get("earnings_estimates", "{}") or "{}")
        except:
            d["earnings"] = {}
        reports.append(d)

    return render_template("compare.html", stocks=stocks, selected=stock, reports=reports)


# ?€?€ ?µê³„ ?˜ì´ì§€ ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
@app.route("/stats")
@login_required
def stats():
    conn = get_db()
    # ?¼ë³„ ?…ë¡œ??    daily = conn.execute("""
        SELECT report_date, COUNT(*) as cnt,
               SUM(CASE WHEN report_type='stock' THEN 1 ELSE 0 END) as stocks,
               SUM(CASE WHEN report_type='industry' THEN 1 ELSE 0 END) as industries,
               SUM(CASE WHEN report_type='strategy' THEN 1 ELSE 0 END) as strategies,
               SUM(CASE WHEN prev_rating != '' THEN 1 ELSE 0 END) as upgrades
        FROM reports GROUP BY report_date ORDER BY report_date DESC LIMIT 30
    """).fetchall()

    # ì¦ê¶Œ?¬ë³„
    by_broker = conn.execute("""
        SELECT brokerage, COUNT(*) as cnt FROM reports
        WHERE brokerage != '' GROUP BY brokerage ORDER BY cnt DESC LIMIT 15
    """).fetchall()

    # ì¢…ëª©ë³?    by_stock = conn.execute("""
        SELECT stock_name, COUNT(*) as cnt, COUNT(DISTINCT brokerage) as brokers
        FROM reports WHERE stock_name != '' AND report_type = 'stock'
        GROUP BY stock_name ORDER BY cnt DESC LIMIT 15
    """).fetchall()

    # ?˜ê²¬ë³€ê²??ˆìŠ¤? ë¦¬
    rating_changes = conn.execute("""
        SELECT report_date, stock_name, brokerage, prev_rating, rating, prev_target_price, target_price
        FROM reports WHERE prev_rating != '' ORDER BY report_date DESC LIMIT 20
    """).fetchall()

    # ì´??µê³„
    total = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
    total_dates = conn.execute("SELECT COUNT(DISTINCT report_date) FROM reports").fetchone()[0]

    conn.close()
    return render_template("stats.html", daily=daily, by_broker=by_broker, by_stock=by_stock,
                          rating_changes=rating_changes, total=total, total_dates=total_dates)


# ?€?€ ?„í´ë¦??œë¨¸ë¦??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
@app.route("/weekly")
@login_required
def weekly_summary():
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        ref = datetime.strptime(date_str, "%Y-%m-%d")
    except:
        ref = datetime.now()

    # ?´ë‹¹ ì£¼ì˜ ??ê¸?ê³„ì‚°
    mon = ref - timedelta(days=ref.weekday())
    fri = mon + timedelta(days=4)
    mon_str = mon.strftime("%Y-%m-%d")
    fri_str = fri.strftime("%Y-%m-%d")

    conn = get_db()

    # ì£¼ê°„ ë¦¬í¬???µê³„
    weekly_total = conn.execute(
        "SELECT COUNT(*) FROM reports WHERE report_date BETWEEN ? AND ?",
        [mon_str, fri_str]).fetchone()[0]

    # ?¼ë³„ ì¹´ìš´??    daily_counts = conn.execute("""
        SELECT report_date, COUNT(*) as cnt,
               SUM(CASE WHEN report_type='stock' THEN 1 ELSE 0 END) as stocks,
               SUM(CASE WHEN report_type='industry' THEN 1 ELSE 0 END) as industries,
               SUM(CASE WHEN report_type='strategy' THEN 1 ELSE 0 END) as strategies
        FROM reports WHERE report_date BETWEEN ? AND ?
        GROUP BY report_date ORDER BY report_date
    """, [mon_str, fri_str]).fetchall()

    # ?¸ê¸° ì¢…ëª© TOP 10
    top_stocks = conn.execute("""
        SELECT stock_name, COUNT(*) as cnt, COUNT(DISTINCT brokerage) as brokers,
               GROUP_CONCAT(DISTINCT brokerage) as broker_list
        FROM reports WHERE report_date BETWEEN ? AND ? AND stock_name != '' AND report_type = 'stock'
        GROUP BY stock_name ORDER BY cnt DESC LIMIT 10
    """, [mon_str, fri_str]).fetchall()

    # ?˜ê²¬ ë³€ê²?ëª©ë¡
    rating_changes = conn.execute("""
        SELECT report_date, stock_name, brokerage, prev_rating, rating,
               prev_target_price, target_price
        FROM reports WHERE report_date BETWEEN ? AND ? AND prev_rating != ''
        ORDER BY report_date DESC
    """, [mon_str, fri_str]).fetchall()

    # ?¹í„° ë¶„í¬
    sector_dist = conn.execute("""
        SELECT sector, COUNT(*) as cnt FROM reports
        WHERE report_date BETWEEN ? AND ? AND sector != '' AND sector IS NOT NULL
        GROUP BY sector ORDER BY cnt DESC LIMIT 10
    """, [mon_str, fri_str]).fetchall()

    # ì¦ê¶Œ?¬ë³„ ?œë™
    broker_activity = conn.execute("""
        SELECT brokerage, COUNT(*) as cnt FROM reports
        WHERE report_date BETWEEN ? AND ? AND brokerage != ''
        GROUP BY brokerage ORDER BY cnt DESC LIMIT 10
    """, [mon_str, fri_str]).fetchall()

    # ì£¼ê°„ ì»¨ì½œ
    concalls = conn.execute("""
        SELECT company_name, concall_date, host, sentiment, keywords, headline
        FROM concalls WHERE concall_date BETWEEN ? AND ?
        ORDER BY concall_date
    """, [mon_str, fri_str]).fetchall()

    # ì»¨ì½œ ?¤ì›Œ???©ì‚°
    all_kw = []
    concall_list = []
    for cc in concalls:
        d = dict(cc)
        try:
            kw_list = json.loads(d.get("keywords", "[]") or "[]")
            all_kw.extend(kw_list)
            d["keyword_list"] = kw_list
        except:
            d["keyword_list"] = []
        concall_list.append(d)

    # ?¤ì›Œ??ë¹ˆë„ ?•ë ¬
    kw_freq = {}
    for kw in all_kw:
        kw_freq[kw] = kw_freq.get(kw, 0) + 1
    kw_sorted = sorted(kw_freq.items(), key=lambda x: -x[1])[:15]

    conn.close()

    prev_week = (mon - timedelta(days=7)).strftime("%Y-%m-%d")
    next_week = (mon + timedelta(days=7)).strftime("%Y-%m-%d")

    return render_template("weekly.html",
        mon=mon, fri=fri, mon_str=mon_str, fri_str=fri_str,
        weekly_total=weekly_total, daily_counts=daily_counts,
        top_stocks=top_stocks, rating_changes=rating_changes,
        sector_dist=sector_dist, broker_activity=broker_activity,
        concall_list=concall_list, kw_sorted=kw_sorted,
        prev_week=prev_week, next_week=next_week,
        date_str=date_str, today=datetime.now().strftime("%Y-%m-%d"),
    )


# ?€?€ ì»¨ì„¼?œìŠ¤ ì¶”ì  ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def _parse_num(s):
    """'12.5ì¡?, '8,500??, '180,000?? ?±ì—???«ì ì¶”ì¶œ (ì¡????¨ìœ„ ?µì¼?’ì–µ)"""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    multiplier = 1
    if "ì¡? in s:
        multiplier = 10000  # ì¡?????        s = s.replace("ì¡?, "")
    elif "?? in s:
        multiplier = 1
        s = s.replace("??, "")
    s = re.sub(r"[^\d.\-]", "", s)
    if not s:
        return None
    try:
        return round(float(s) * multiplier, 1)
    except:
        return None


@app.route("/consensus")
@login_required
def consensus():
    stock = request.args.get("stock", "")
    conn = get_db()

    # ì¢…ëª© ëª©ë¡ (2ê±??´ìƒ ë¦¬í¬??
    stocks = conn.execute("""
        SELECT stock_name, COUNT(*) as cnt, COUNT(DISTINCT brokerage) as brokers,
               MIN(report_date) as first_date, MAX(report_date) as last_date
        FROM reports WHERE stock_name != '' AND report_type = 'stock' AND target_price != ''
        GROUP BY stock_name HAVING cnt >= 2 ORDER BY cnt DESC LIMIT 30
    """).fetchall()

    history = []
    chart_data = []
    earnings_chart = []
    current_consensus = {}
    if stock:
        rows = conn.execute("""
            SELECT report_date, brokerage, target_price, rating, analyst,
                   prev_target_price, prev_rating, earnings_estimates
            FROM reports WHERE stock_name = ? AND report_type = 'stock'
            ORDER BY report_date ASC, brokerage
        """, [stock]).fetchall()
        history = [dict(r) for r in rows]

        # ?€?€ ëª©í‘œê°€ ì°¨íŠ¸ ?°ì´???€?€
        date_prices = {}
        for h in history:
            tp = h.get("target_price", "")
            nums = re.sub(r"[^\d]", "", tp)
            if nums:
                price = int(nums)
                d = h["report_date"]
                if d not in date_prices:
                    date_prices[d] = []
                date_prices[d].append(price)

        all_prices_so_far = []
        for d in sorted(date_prices.keys()):
            all_prices_so_far.extend(date_prices[d])
            avg = sum(date_prices[d]) / len(date_prices[d])
            cumul_avg = sum(all_prices_so_far) / len(all_prices_so_far)
            chart_data.append({
                "date": d,
                "avg_tp": round(avg),
                "cumul_avg": round(cumul_avg),
                "count": len(date_prices[d])
            })

        # ?€?€ ?¤ì ì¶”ì •ì¹?ë³€???°ì´???€?€
        # ê°€??ê°€ê¹Œìš´ ?°ë„(?¬í•´+1) ê¸°ì??¼ë¡œ ë§¤ì¶œ/?ì—…?´ìµ ì¶”ì •ì¹?ë³€??ì¶”ì 
        from datetime import date as dt_date
        next_year = f"{dt_date.today().year}E"  # ?? "2026E"
        this_year = f"{dt_date.today().year - 1}E"  # ?? "2025E" (ì§ì „?„ë„ ?¤ì )

        date_earnings = {}  # {date: [{"revenue":X, "op":Y}]}
        for h in history:
            try:
                ee = json.loads(h.get("earnings_estimates", "{}") or "{}")
            except:
                continue
            if not ee:
                continue
            # next_year ?°ì„ , ?†ìœ¼ë©?this_year
            yr_data = ee.get(next_year) or ee.get(this_year) or {}
            rev = _parse_num(yr_data.get("revenue", ""))
            op = _parse_num(yr_data.get("op", ""))
            if rev or op:
                d = h["report_date"]
                if d not in date_earnings:
                    date_earnings[d] = []
                date_earnings[d].append({"revenue": rev, "op": op, "brokerage": h.get("brokerage", "")})

        # ? ì§œë³??‰ê·  ë§¤ì¶œ/?ì—…?´ìµ ì¶”ì •ì¹?        for d in sorted(date_earnings.keys()):
            items = date_earnings[d]
            revs = [x["revenue"] for x in items if x["revenue"]]
            ops = [x["op"] for x in items if x["op"]]
            entry = {"date": d, "count": len(items)}
            entry["avg_rev"] = round(sum(revs) / len(revs), 1) if revs else None
            entry["avg_op"] = round(sum(ops) / len(ops), 1) if ops else None
            earnings_chart.append(entry)

        # ë³€?”ìœ¨ ê³„ì‚° (ì²??°ì´???€ë¹?
        if earnings_chart:
            base_rev = next((e["avg_rev"] for e in earnings_chart if e["avg_rev"]), None)
            base_op = next((e["avg_op"] for e in earnings_chart if e["avg_op"]), None)
            for e in earnings_chart:
                if base_rev and e["avg_rev"]:
                    e["rev_chg"] = round((e["avg_rev"] - base_rev) / base_rev * 100, 1)
                else:
                    e["rev_chg"] = None
                if base_op and e["avg_op"]:
                    e["op_chg"] = round((e["avg_op"] - base_op) / base_op * 100, 1)
                else:
                    e["op_chg"] = None

        # ?€?€ ìµœì‹  ì»¨ì„¼?œìŠ¤ ?€?€
        latest_by_broker = {}
        for h in history:
            latest_by_broker[h["brokerage"]] = h

        prices_for_avg = []
        for b, h in latest_by_broker.items():
            tp = h.get("target_price", "")
            nums = re.sub(r"[^\d]", "", tp)
            if nums:
                prices_for_avg.append(int(nums))
        current_consensus = {
            "brokers": latest_by_broker,
            "avg_tp": round(sum(prices_for_avg) / len(prices_for_avg)) if prices_for_avg else 0,
            "high": max(prices_for_avg) if prices_for_avg else 0,
            "low": min(prices_for_avg) if prices_for_avg else 0,
            "count": len(prices_for_avg),
        }

    conn.close()
    return render_template("consensus.html",
        stocks=stocks, selected=stock, history=history,
        chart_data=chart_data, earnings_chart=earnings_chart,
        current_consensus=current_consensus,
    )


# ?€?€ ì»¨ì½œ ?¸íŠ¸ ê¸°ëŠ¥ ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def fetch_clova_transcript(url):
    """?´ë¡œë°”ë…¸??ê³µìœ  ë§í¬?ì„œ ?ìŠ¤??ì¶”ì¶œ"""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        os.system("pip install beautifulsoup4 --break-system-packages -q")
        from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://clovanote.naver.com/",
    }

    # ê³µìœ  ID ì¶”ì¶œ
    share_id = url.rstrip("/").split("/")[-1]
    logger.info(f"?” ?´ë¡œë°”ë…¸??share_id: {share_id}")

    text_parts = []

    # ?€?€ ë°©ë²•1: ê³µìœ  API ì§ì ‘ ?¸ì¶œ ?€?€
    api_urls = [
        f"https://clovanote.naver.com/api/v1/share/{share_id}",
        f"https://clovanote.naver.com/api/v2/share/{share_id}",
        f"https://clovanote.naver.com/api/share/{share_id}",
    ]
    api_headers = {**headers, "Accept": "application/json"}

    for api_url in api_urls:
        try:
            resp = httpx.get(api_url, headers=api_headers, timeout=15, follow_redirects=True)
            if resp.status_code == 200:
                data = resp.json()
                # ?¤ì–‘??JSON êµ¬ì¡°?ì„œ ?ìŠ¤??ì¶”ì¶œ
                text_parts = _extract_text_from_json(data)
                if text_parts:
                    logger.info(f"??API?ì„œ ?ìŠ¤??ì¶”ì¶œ ?±ê³µ: {len(text_parts)}ê°??¸ê·¸ë¨¼íŠ¸")
                    break
        except Exception as e:
            logger.debug(f"API {api_url} ?¤íŒ¨: {e}")
            continue

    # ?€?€ ë°©ë²•2: HTML ?˜ì´ì§€?ì„œ __NEXT_DATA__ ì¶”ì¶œ (ì¿ í‚¤ ?¬í•¨) ?€?€
    if not text_parts:
        try:
            cookies_dict = {}
            try:
                import browser_cookie3
                cj = browser_cookie3.chrome(domain_name=".naver.com")
                for c in cj:
                    cookies_dict[c.name] = c.value
                logger.info(f"?ª ?¤ì´ë²?ì¿ í‚¤ {len(cookies_dict)}ê°?ë¡œë“œ")
            except Exception as e:
                logger.warning(f"ì¿ í‚¤ ë¡œë“œ ?¤íŒ¨: {e}")

            resp = httpx.get(url, headers=headers, timeout=30, follow_redirects=True, cookies=cookies_dict)
            soup = BeautifulSoup(resp.text, "html.parser")

            # __NEXT_DATA__ (Next.js SSR ?°ì´??
            next_data = soup.find("script", id="__NEXT_DATA__")
            if next_data and next_data.string:
                try:
                    nd = json.loads(next_data.string)
                    text_parts = _extract_text_from_json(nd)
                    if text_parts:
                        logger.info(f"??__NEXT_DATA__?ì„œ ?ìŠ¤??ì¶”ì¶œ ?±ê³µ")
                except:
                    pass

            # script ?œê·¸ ??JSON ?°ì´??            if not text_parts:
                for script in soup.find_all("script"):
                    st = script.string or ""
                    if any(kw in st for kw in ["noteText", "segments", "transcript", "sttText", "content"]):
                        # JSON ë¸”ë¡ ì¶”ì¶œ
                        json_matches = re.findall(r'\{[^{}]{50,}\}', st)
                        for jm in json_matches:
                            try:
                                d = json.loads(jm)
                                parts = _extract_text_from_json(d)
                                if parts:
                                    text_parts.extend(parts)
                            except:
                                pass
                        # ì§ì ‘ ?ìŠ¤???¨í„´
                        if not text_parts:
                            for pat in [r'"text"\s*:\s*"((?:[^"\\]|\\.){10,})"',
                                       r'"noteText"\s*:\s*"((?:[^"\\]|\\.){10,})"',
                                       r'"sttText"\s*:\s*"((?:[^"\\]|\\.){10,})"',
                                       r'"content"\s*:\s*"((?:[^"\\]|\\.){10,})"']:
                                matches = re.findall(pat, st)
                                text_parts.extend(matches)

            # DOM ?”ì†Œ?ì„œ ì¶”ì¶œ
            if not text_parts:
                for sel in ["[class*='segment']", "[class*='text']", "[class*='content']",
                           "[class*='note']", "[data-testid]", "article", "main p"]:
                    for el in soup.select(sel):
                        t = el.get_text(strip=True)
                        if len(t) > 30:
                            text_parts.append(t)
                    if text_parts:
                        break

            # ìµœí›„?˜ë‹¨: body ?„ì²´ ?ìŠ¤??            if not text_parts:
                body = soup.find("body")
                if body:
                    raw = body.get_text(separator="\n", strip=True)
                    # ?ˆë¬´ ì§§ìœ¼ë©?(JS ?Œë”ë§??„ìš”) ?¤íŒ¨ ì²˜ë¦¬
                    if len(raw) > 200:
                        text_parts = [raw]

        except Exception as e:
            logger.error(f"HTML ?Œì‹± ?¤íŒ¨: {e}")

    # ?€?€ ë°©ë²•3: Selenium fallback ?€?€
    if not text_parts:
        text_parts = _fetch_clova_selenium(url)

    full_text = "\n".join(text_parts)
    lines = list(dict.fromkeys(full_text.split("\n")))
    full_text = "\n".join(l for l in lines if l.strip())

    logger.info(f"?“ ?´ë¡œë°”ë…¸???ìŠ¤??ê¸¸ì´: {len(full_text)}??)
    return full_text[:15000] if len(full_text) > 15000 else full_text


def _extract_text_from_json(data, depth=0):
    """JSON?ì„œ ?ìŠ¤???¸ê·¸ë¨¼íŠ¸ ?¬ê? ì¶”ì¶œ"""
    if depth > 10:
        return []
    parts = []
    if isinstance(data, dict):
        # ì§ì ‘ ?ìŠ¤???„ë“œ
        for key in ["text", "noteText", "sttText", "content", "transcript", "summary"]:
            v = data.get(key)
            if isinstance(v, str) and len(v) > 20:
                parts.append(v)
        # segments/sections ë°°ì—´
        for key in ["segments", "sections", "notes", "paragraphs", "results", "items", "data", "props"]:
            v = data.get(key)
            if isinstance(v, (list, dict)):
                parts.extend(_extract_text_from_json(v, depth + 1))
        # pageProps ??Next.js êµ¬ì¡°
        for key in ["pageProps", "dehydratedState", "queries"]:
            v = data.get(key)
            if isinstance(v, dict):
                parts.extend(_extract_text_from_json(v, depth + 1))
    elif isinstance(data, list):
        for item in data:
            parts.extend(_extract_text_from_json(item, depth + 1))
    return parts


NAVER_COOKIES_FILE = Path(__file__).parent / "naver_cookies.json"


def _save_naver_cookies(cookies):
    """ì¿ í‚¤ ?€??""
    NAVER_COOKIES_FILE.write_text(json.dumps(cookies, ensure_ascii=False), encoding="utf-8")
    logger.info(f"?ª ?¤ì´ë²?ì¿ í‚¤ {len(cookies)}ê°??€??)


def _load_naver_cookies():
    """?€?¥ëœ ì¿ í‚¤ ë¡œë“œ"""
    if NAVER_COOKIES_FILE.exists():
        try:
            cookies = json.loads(NAVER_COOKIES_FILE.read_text(encoding="utf-8"))
            logger.info(f"?ª ?¤ì´ë²?ì¿ í‚¤ {len(cookies)}ê°?ë¡œë“œ")
            return cookies
        except:
            pass
    return []


def _get_chrome_service():
    """ChromeDriver ?œë¹„??""
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        return Service(ChromeDriverManager().install())
    except:
        from selenium.webdriver.chrome.service import Service
        return Service()


def naver_login_interactive():
    """Chrome ì°??„ì›Œ???¤ì´ë²?ë¡œê·¸????ì¿ í‚¤ ?€??(ìµœì´ˆ 1??"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        return False

    opts = Options()
    opts.add_argument("--window-size=500,700")
    opts.add_argument("--disable-gpu")

    driver = None
    try:
        driver = webdriver.Chrome(service=_get_chrome_service(), options=opts)
        driver.get("https://nid.naver.com/nidlogin.login?url=https%3A%2F%2Fclovanote.naver.com")
        logger.info("?Œ ?¤ì´ë²?ë¡œê·¸??ì°??´ë¦¼ - ë¡œê·¸???€ê¸?ì¤?..")

        # ë¡œê·¸???„ë£Œ ?€ê¸?(ìµœë? 120ì´?
        for _ in range(120):
            time.sleep(1)
            cur = driver.current_url
            if "nidlogin" not in cur and "nid.naver.com" not in cur:
                break

        time.sleep(2)
        if "clovanote" not in driver.current_url:
            driver.get("https://clovanote.naver.com")
            time.sleep(3)

        cookies = driver.get_cookies()
        _save_naver_cookies(cookies)
        driver.quit()
        return len(cookies) > 0

    except Exception as e:
        logger.error(f"???¤ì´ë²?ë¡œê·¸???¤íŒ¨: {e}")
        if driver:
            try: driver.quit()
            except: pass
        return False


def _fetch_clova_selenium(url):
    """Selenium + ?€?¥ëœ ì¿ í‚¤ë¡??´ë¡œë°”ë…¸???ìŠ¤??ì¶”ì¶œ"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
    except ImportError:
        logger.warning("? ï¸ Selenium ë¯¸ì„¤ì¹?)
        return []

    cookies = _load_naver_cookies()
    if not cookies:
        logger.warning("? ï¸ ?€?¥ëœ ?¤ì´ë²?ì¿ í‚¤ ?†ìŒ. /login_naver ?¤í–‰ ?„ìš”")
        return []

    logger.info(f"?Œ Selenium?¼ë¡œ ?´ë¡œë°”ë…¸??ë¡œë”©: {url}")
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")

    driver = None
    try:
        driver = webdriver.Chrome(service=_get_chrome_service(), options=opts)
        driver.set_page_load_timeout(30)

        # ì¿ í‚¤ ì£¼ì…: ë¨¼ì? ?„ë©”???‘ì†
        driver.get("https://clovanote.naver.com")
        time.sleep(2)
        for c in cookies:
            try:
                cookie = {"name": c["name"], "value": c["value"]}
                if "domain" in c: cookie["domain"] = c["domain"]
                if "path" in c: cookie["path"] = c["path"]
                driver.add_cookie(cookie)
            except:
                pass
        logger.info("?ª ì¿ í‚¤ ì£¼ì… ?„ë£Œ")

        # ?¤ì œ URL ë¡œë”©
        driver.get(url)

        # ?˜ì´ì§€ ë¡œë”© ?€ê¸?        time.sleep(10)

        # ?¤í¬ë¡¤ë¡œ lazy load ?¸ë¦¬ê±?        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)

        # ?”ë²„ê·? ?˜ì´ì§€ ?ŒìŠ¤ ?€??        try:
            debug_path = Path(__file__).parent / "debug_clova.html"
            debug_path.write_text(driver.page_source, encoding="utf-8")
            logger.info(f"?“„ ?”ë²„ê·??˜ì´ì§€ ?€?? {debug_path}")
        except:
            pass

        # ë¡œê·¸???˜ì´ì§€ ì²´í¬
        if "nidlogin" in driver.current_url or "?¤ì´ë²?: ë¡œê·¸?? in driver.title:
            logger.warning("? ï¸ ì¿ í‚¤ ë§Œë£Œ - /login_naver ?¬ì‹¤???„ìš”")
            driver.quit()
            return []

        parts = []

        # 1?œìœ„: ?´ë¡œë°”ë…¸???ìŠ¤???”ì†Œ
        selectors = [
            "[class*='segment'] [class*='text']",
            "[class*='segment']",
            "[class*='stt']",
            "[class*='transcript']",
            "[class*='note_text']",
            "[class*='NoteText']",
            "[class*='content_text']",
            "[class*='voice_text']",
            "[data-testid*='segment']",
            "[data-testid*='text']",
            "article p", "main p",
        ]
        for sel in selectors:
            try:
                elems = driver.find_elements(By.CSS_SELECTOR, sel)
                for el in elems:
                    t = el.text.strip()
                    if len(t) > 10:
                        parts.append(t)
                if parts:
                    logger.info(f"???€?‰í„° '{sel}'?ì„œ {len(parts)}ê°??ìŠ¤??)
                    break
            except:
                continue

        # 2?œìœ„: ?˜ì´ì§€ ?ŒìŠ¤?ì„œ JSON ì¶”ì¶œ
        if not parts:
            page_source = driver.page_source
            json_patterns = [
                r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                r'"segments"\s*:\s*(\[.*?\])',
                r'"noteText"\s*:\s*"((?:[^"\\]|\\.){20,})"',
                r'"text"\s*:\s*"((?:[^"\\]|\\.){30,})"',
            ]
            for pat in json_patterns:
                matches = re.findall(pat, page_source, re.DOTALL)
                for m in matches:
                    try:
                        d = json.loads(m)
                        extracted = _extract_text_from_json(d)
                        if extracted:
                            parts.extend(extracted)
                            break
                    except:
                        if isinstance(m, str) and len(m) > 30:
                            parts.append(m)
                if parts:
                    break

        # 3?œìœ„: body ?„ì²´ ?ìŠ¤??        if not parts:
            try:
                body_text = driver.find_element(By.TAG_NAME, "body").text
                lines = [l.strip() for l in body_text.split("\n") if len(l.strip()) > 15]
                if len(lines) > 5:
                    parts = lines
            except:
                pass

        driver.quit()
        return parts

    except Exception as e:
        logger.error(f"??Selenium ?¤íŒ¨: {e}")
        if driver:
            try: driver.quit()
            except: pass
        return []



def parse_concall(transcript, company_name="", host=""):
    """ì»¨ì½œ ?ìŠ¤?¸ë? Geminië¡??”ì•½"""
    if not transcript.strip():
        raise Exception("?ìŠ¤??ì¶”ì¶œ ?¤íŒ¨")

    prompt = f"""You are a Korean equity research analyst. Summarize this conference call transcript.

Company: {company_name}
Host: {host}

RULES:
- ALL output MUST be in Korean
- Focus on: ?¤ì , ê°€?´ë˜?? ?¬ì—… ?„ëµ, ?„í—˜ ?”ì¸
- Extract specific numbers/percentages
- Q&A ?µì‹¬ ?´ìš©??ë°˜ë“œ???¬í•¨
- ê°€?´ë˜???¤ì  ?„ë§ ?˜ì¹˜??ë°˜ë“œ??ì¶”ì¶œ

Return ONLY valid JSON:

{{
  "company_name": "ì¢…ëª©ëª??œêµ­??",
  "headline": "?µì‹¬ ??ì¤??”ì•½ (50???´ë‚´). ê°€??ì¤‘ìš”??ê²°ë¡ /?œí”„?¼ì´ì¦?,
  "key_points": ["?µì‹¬ 5~8ê°? êµ¬ì²´???˜ì¹˜ ?„ìˆ˜. ?¤ì , ?„ëµ, ê°€?´ë˜??ì¤‘ì‹¬"],
  "guidance": "?Œì‚¬ ê°€?´ë˜???„ë§ ?”ì•½ ?ìŠ¤??2~3ì¤?,
  "guidance_numbers": [
    {{"item": "??ª©ëª??? 2026 ë§¤ì¶œ, 2026 ?ì—…?´ìµ, CAPEX ??", "value": "?˜ì¹˜+?¨ìœ„", "direction": "up/down/flat", "detail": "?„ë…„ë¹?+15% ??ë¶€ê°€?¤ëª…"}}
  ],
  "earnings_surprise": {{
    "quarter": "?´ë‹¹ ë¶„ê¸° (?? 4Q25)",
    "items": [
      {{"item": "??ª©(ë§¤ì¶œ/?ì—…?´ìµ/?œì´??", "consensus": "ì»¨ì„¼?œìŠ¤ ?˜ì¹˜", "actual": "?¤ì œ ?˜ì¹˜", "beat": "beat/miss/inline"}}
    ]
  }},
  "qa_highlights": ["Q&A ?µì‹¬ 3~5ê°? ì§ˆë¬¸ ë§¥ë½ + ?µë? ?”ì?"],
  "keywords": ["ì»¨ì½œ?ì„œ ?ì£¼ ?¸ê¸‰???µì‹¬ ?¤ì›Œ???Œë§ˆ 5~10ê°? ?? AI, HBM, ?„ë ¥ë°˜ë„ì²? CAPEX, ?˜ì£¼?”ê³ "],
  "sentiment": "positive/neutral/negative - ?„ì²´ ??,
  "summary": "3~4ë¬¸ì¥ ì¢…í•© ?”ì•½. ?¬ì?ê? 1ë¶??ˆì— ?Œì•… ê°€?¥í•˜?„ë¡"
}}

ONLY JSON. No markdown.

TRANSCRIPT:
{transcript[:12000]}"""

    for model in GROQ_MODELS:
        wait_rate()
        try:
            logger.info(f"Concall trying: {model}")
            resp = httpx.post(GROQ_API_URL, headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }, json={"model": model, "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2, "max_tokens": 2000}, timeout=60)

            if resp.status_code == 429:
                logger.warning(f"429 rate limit: {model}")
                time.sleep(30)
                continue
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            text = re.sub(r"```json\s*", "", text)
            text = re.sub(r"```\s*$", "", text)
            data = json.loads(text.strip())
            logger.info(f"??Concall parsed with {model}")
            return data
        except json.JSONDecodeError:
            continue
        except Exception as e:
            logger.error(f"Concall error {model}: {e}")
            continue

    raise Exception("ëª¨ë“  ëª¨ë¸ ?œë„ ?¤íŒ¨")


def save_concall(data, clova_url="", concall_time="", concall_date="", host="", transcript=""):
    conn = get_db()
    # ì»¨ì½œ ì¤‘ë³µ ì²´í¬ (ê°™ì? clova_url?´ë©´ ?¤í‚µ)
    if clova_url:
        existing = conn.execute("SELECT id FROM concalls WHERE clova_url = ?", [clova_url]).fetchone()
        if existing:
            logger.info(f"ì»¨ì½œ ì¤‘ë³µ ?¤í‚µ: {clova_url}")
            conn.close()
            return existing[0]
    kp = json.dumps(data.get("key_points", []), ensure_ascii=False)
    qa = json.dumps(data.get("qa_highlights", []), ensure_ascii=False)
    gn = json.dumps(data.get("guidance_numbers", []), ensure_ascii=False)
    es = json.dumps(data.get("earnings_surprise", {}), ensure_ascii=False)
    kw = json.dumps(data.get("keywords", []), ensure_ascii=False)
    if not concall_date:
        concall_date = datetime.now().strftime("%Y-%m-%d")
    conn.execute("""
        INSERT INTO concalls (company_name, host, concall_time, concall_date, clova_url,
            transcript, headline, key_points, guidance, guidance_numbers, earnings_surprise,
            qa_highlights, keywords, sentiment, summary)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data.get("company_name", ""),
        host,
        concall_time,
        concall_date,
        clova_url,
        transcript[:5000],
        data.get("headline", ""),
        kp, data.get("guidance", ""), gn, es, qa, kw,
        data.get("sentiment", "neutral"),
        data.get("summary", ""),
    ))
    conn.commit()
    rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    # ?¤ì‹œê°??Œë¦¼
    try:
        sent = data.get("sentiment", "neutral")
        se = {"positive": "?Ÿ¢", "negative": "?”´"}.get(sent, "?Ÿ¡")
        msg = f"?™ï¸?*??ì»¨í¼?°ìŠ¤ì½??±ë¡*\n"
        msg += f"{se} *{data.get('company_name', '')}*"
        if host:
            msg += f" ({host})"
        msg += "\n"
        if data.get("headline"):
            msg += f"?’¡ {data['headline']}\n"
        dash_url = get_dashboard_url()
        msg += f"?”— [ì»¨ì½œ ?ì„¸]({dash_url}/concalls)"
        notify_all_chats(msg)
    except Exception as e:
        logger.error(f"ì»¨ì½œ ?Œë¦¼ ?¤íŒ¨: {e}")

    return rid


# ?€?€ ì»¨ì½œ ?€?œë³´???¼ìš°???€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
@app.route("/concalls")
@login_required
def concalls_dashboard():
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    conn = get_db()
    concalls = conn.execute(
        "SELECT * FROM concalls WHERE concall_date = ? ORDER BY concall_time, created_at",
        [date_str]).fetchall()

    result = []
    for c in concalls:
        d = dict(c)
        try:
            d["key_points_list"] = json.loads(d.get("key_points", "[]"))
        except:
            d["key_points_list"] = []
        try:
            d["qa_list"] = json.loads(d.get("qa_highlights", "[]"))
        except:
            d["qa_list"] = []
        try:
            d["guidance_nums"] = json.loads(d.get("guidance_numbers", "[]") or "[]")
        except:
            d["guidance_nums"] = []
        try:
            d["surprise"] = json.loads(d.get("earnings_surprise", "{}") or "{}")
        except:
            d["surprise"] = {}
        try:
            d["keyword_list"] = json.loads(d.get("keywords", "[]") or "[]")
        except:
            d["keyword_list"] = []
        result.append(d)

    total = len(result)

    # ?¤ë¥¸ ? ì§œ???°ì´???ˆëŠ”ì§€
    prev_exists = conn.execute(
        "SELECT COUNT(*) FROM concalls WHERE concall_date < ?", [date_str]).fetchone()[0]
    next_exists = conn.execute(
        "SELECT COUNT(*) FROM concalls WHERE concall_date > ?", [date_str]).fetchone()[0]
    conn.close()

    try:
        current_date = datetime.strptime(date_str, "%Y-%m-%d")
    except:
        current_date = datetime.now()

    prev_date = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
    next_date = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")

    return render_template("concalls.html", concalls=result, date=date_str, total=total,
                          prev_date=prev_date, next_date=next_date, today=today,
                          prev_exists=prev_exists, next_exists=next_exists)


@app.route("/concall-delete/<int:cid>", methods=["POST"])
@login_required
def concall_delete(cid):
    conn = get_db()
    row = conn.execute("SELECT concall_date FROM concalls WHERE id = ?", [cid]).fetchone()
    date = row["concall_date"] if row else datetime.now().strftime("%Y-%m-%d")
    conn.execute("DELETE FROM concalls WHERE id = ?", [cid])
    conn.commit()
    conn.close()
    return redirect(f"/concalls?date={date}")


@app.route("/concall-delete-all", methods=["POST"])
@login_required
def concall_delete_all():
    date_str = request.form.get("date", "")
    conn = get_db()
    if date_str:
        conn.execute("DELETE FROM concalls WHERE concall_date = ?", [date_str])
    conn.commit()
    conn.close()
    return redirect(f"/concalls?date={date_str}" if date_str else "/concalls")


# ?€?€ ?¸ë? URL ?¬í¼ ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def get_dashboard_url():
    """ngrok URL ?ëŠ” ë¡œì»¬ URL ë°˜í™˜"""
    global NGROK_URL
    try:
        resp = httpx.get("http://127.0.0.1:4040/api/tunnels", timeout=3)
        for t in resp.json().get("tunnels", []):
            if t.get("proto") == "https":
                NGROK_URL = t["public_url"]
                return f"{NGROK_URL}/dashboard"
    except:
        pass
    if NGROK_URL:
        return f"{NGROK_URL}/dashboard"
    return "http://localhost:5000/dashboard"


# ?€?€ ?”ë ˆê·¸ë¨ ë´??€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def run_telegram_bot():
    if not TELEGRAM_TOKEN:
        logger.info("?”ë ˆê·¸ë¨ ë´?ë¹„í™œ?±í™” (? í° ?†ìŒ)")
        return
    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, MessageHandler, filters
        import asyncio
    except ImportError:
        logger.warning("python-telegram-bot ë¯¸ì„¤ì¹?)
        return

    # PDF ì²˜ë¦¬ ??    import queue as _queue
    pdf_queue = _queue.Queue()

    async def process_one(bot, chat_id, msg_id, pdf_bytes, safe_name, file_name):
        """PDF 1ê°?ì²˜ë¦¬"""
        try:
            loop = asyncio.get_event_loop()
            parsed = await loop.run_in_executor(None, parse_pdf, bytes(pdf_bytes))
            rid = await loop.run_in_executor(None, save_report, parsed, safe_name)
            if rid is None:
                # ì¤‘ë³µ ë¦¬í¬??                await bot.edit_message_text(
                    chat_id=chat_id, message_id=msg_id,
                    text=f"? ï¸ ì¤‘ë³µ ë¦¬í¬?? {parsed.get('stock_name','')} - {parsed.get('brokerage','')}\n?´ë? ?±ë¡??ë¦¬í¬?¸ì…?ˆë‹¤."
                )
                return
            await loop.run_in_executor(None, save_to_notion, parsed)

            emoji = {"Buy": "?Ÿ¢", "Overweight": "?Ÿ¢", "Outperform": "?Ÿ¢",
                    "Neutral": "?Ÿ¡", "Hold": "?Ÿ¡", "Sell": "?”´", "Underweight": "?”´"
                    }.get(parsed.get("rating", ""), "??)
            type_emoji = {"stock": "?“Š", "industry": "?­", "strategy": "?Œ"}.get(parsed.get("report_type", ""), "?“„")
            type_kr = {"stock": "ì¢…ëª©", "industry": "?°ì—…", "strategy": "?„ëµ"}.get(parsed.get("report_type", ""), "")

            rating_str = f"{emoji} {parsed.get('rating', 'N/A')}"
            if parsed.get("prev_rating"):
                rating_str = f"?”„ {parsed['prev_rating']} ??*{parsed.get('rating', '')}*"
            tp_str = f"?¯ {parsed.get('target_price', 'N/A')}"
            if parsed.get("prev_target_price"):
                tp_str = f"?¯ {parsed['prev_target_price']} ??*{parsed.get('target_price', '')}*"
            kp = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(parsed.get("key_points", [])[:4]))
            dash_url = get_dashboard_url()

            text = f"??*{parsed.get('stock_name', 'N/A')}* {type_emoji}{type_kr}\n"
            text += f"?â”?â”?â”?â”?â”?â”?â”?â”\n"
            text += f"?¦ {parsed.get('brokerage', '')} Â· {parsed.get('analyst', '')}\n"
            text += f"{rating_str} | {tp_str}\n"
            if parsed.get("headline"):
                text += f"\n?’¡ *{parsed['headline']}*\n"
            if kp:
                text += f"\n{kp}\n"
            if parsed.get("mentioned_stocks"):
                text += f"\n?“Œ {parsed['mentioned_stocks']}\n"
            if parsed.get("risks"):
                text += f"\n? ï¸ {parsed['risks']}\n"
            text += f"\n?”— [?€?œë³´??ë³´ê¸°]({dash_url})"

            await bot.edit_message_text(chat_id=chat_id, message_id=msg_id,
                                       text=text, disable_web_page_preview=True)
        except Exception as e:
            await bot.edit_message_text(chat_id=chat_id, message_id=msg_id,
                                       text=f"??*{file_name}*\n?¤ë¥˜: {str(e)[:200]}")

    async def queue_worker(bot):
        """?ì—???˜ë‚˜??êº¼ë‚´??ì²˜ë¦¬ (30ì´?ê°„ê²©)"""
        while True:
            if not pdf_queue.empty():
                item = pdf_queue.get()
                remaining = pdf_queue.qsize()
                # ì²˜ë¦¬ ?œì‘ ?Œë¦¼
                try:
                    await bot.edit_message_text(
                        chat_id=item["chat_id"], message_id=item["msg_id"],
                        text=f"??*{item['file_name']}*\në¶„ì„ ì¤?..{f' (?€ê¸?{remaining}ê±?' if remaining else ''}")
                except:
                    pass
                await process_one(bot, item["chat_id"], item["msg_id"],
                                 item["pdf_bytes"], item["safe_name"], item["file_name"])
                # ?¤ìŒ ì²˜ë¦¬ ??30ì´??€ê¸?(Gemini ?ë„?œí•œ ë°©ì?)
                if not pdf_queue.empty():
                    await asyncio.sleep(30)
            else:
                await asyncio.sleep(1)

    async def handle_doc(update: Update, context):
        save_chat_id(update.message.chat_id)
        doc = update.message.document
        if not doc.file_name.lower().endswith(".pdf"):
            await update.message.reply_text("? ï¸ PDFë§?ì²˜ë¦¬ ê°€?¥í•©?ˆë‹¤.")
            return

        # PDF ?¤ìš´ë¡œë“œ
        try:
            file = await context.bot.get_file(doc.file_id)
            pdf_bytes = await file.download_as_bytearray()
        except Exception as e:
            if "too big" in str(e).lower() or "file is too big" in str(e).lower():
                await update.message.reply_text(f"? ï¸ *{doc.file_name}*\n?Œì¼???ˆë¬´ ?½ë‹ˆ??(20MB ?œí•œ)")
            else:
                await update.message.reply_text(f"???¤ìš´ë¡œë“œ ?¤íŒ¨: {str(e)[:100]}")
            return

        safe_name = make_safe_filename(doc.file_name)
        (UPLOAD_DIR / safe_name).write_bytes(pdf_bytes)

        # ?ì— ì¶”ê?
        waiting = pdf_queue.qsize()
        if waiting > 0:
            msg = await update.message.reply_text(
                f"?“¥ *{doc.file_name}*\n???€ê¸?ì¤?({waiting}ê±??ì— ì²˜ë¦¬ ì¤?\n_?œì„œ?€ë¡??ë™ ì²˜ë¦¬?©ë‹ˆ??")
        else:
            msg = await update.message.reply_text(
                f"?“¥ *{doc.file_name}*\n??ë¶„ì„ ì¤?..")

        pdf_queue.put({
            "chat_id": update.message.chat_id,
            "msg_id": msg.message_id,
            "pdf_bytes": bytes(pdf_bytes),
            "safe_name": safe_name,
            "file_name": doc.file_name,
        })

    async def start_cmd(update: Update, context):
        save_chat_id(update.message.chat_id)
        dash_url = get_dashboard_url()
        text = ("?“Š *ì¦ê¶Œ??ë¦¬í¬???€?œë³´??ë´?\n\n"
                "?“„ *PDF* ??ë¦¬í¬??ë¶„ì„\n"
                "?”— *?´ë¡œë°”ë…¸??URL* ??ì»¨ì½œ ?”ì•½\n\n"
                "?“Œ *ëª…ë ¹??\n"
                "/start - ?œì‘\n"
                "/today - ?¤ëŠ˜ ë¸Œë¦¬??n"
                "/stats - ?µê³„\n"
                "/login\\_naver - ?¤ì´ë²?ë¡œê·¸??(ìµœì´ˆ 1??\n"
                "/test\\_clova - ?´ë¡œë°”ë…¸??ì¶”ì¶œ ?ŒìŠ¤??n"
                f"\n?”— [ë¦¬í¬???€?œë³´??({dash_url})"
                f"\n?™ï¸?[ì»¨ì½œ ?€?œë³´??({dash_url.replace('/dashboard', '/concalls')})")
        await update.message.reply_text(text, disable_web_page_preview=True)

    async def today_cmd(update: Update, context):
        """?¤ëŠ˜ ë¸Œë¦¬??""
        today = datetime.now().strftime("%Y-%m-%d")
        briefing = generate_briefing(today)
        await update.message.reply_text(briefing, disable_web_page_preview=True)

    async def stats_cmd(update: Update, context):
        """?µê³„ ?”ì•½"""
        conn = get_db()
        total = conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        today_cnt = conn.execute("SELECT COUNT(*) FROM reports WHERE report_date = ?",
                                [datetime.now().strftime("%Y-%m-%d")]).fetchone()[0]
        upgrades = conn.execute("SELECT COUNT(*) FROM reports WHERE prev_rating != ''").fetchone()[0]
        top_stocks = conn.execute("""
            SELECT stock_name, COUNT(*) as cnt FROM reports
            WHERE stock_name != '' AND report_type = 'stock'
            GROUP BY stock_name ORDER BY cnt DESC LIMIT 5
        """).fetchall()
        conn.close()

        dash_url = get_dashboard_url()
        text = f"?“ˆ *?µê³„*\n?â”?â”?â”?â”?â”?â”?â”?â”\n"
        text += f"?„ì²´: {total}ê±?| ?¤ëŠ˜: {today_cnt}ê±?n"
        text += f"?˜ê²¬ë³€ê²? {upgrades}ê±?n\n"
        if top_stocks:
            text += "?† *ë§ì´ ì»¤ë²„??ì¢…ëª©*\n"
            for s in top_stocks:
                text += f"  Â· {s['stock_name']} ({s['cnt']}ê±?\n"
        text += f"\n?”— [?ì„¸ ?µê³„]({dash_url.replace('/dashboard', '/stats')})"
        await update.message.reply_text(text, disable_web_page_preview=True)

    async def test_clova_cmd(update: Update, context):
        """?´ë¡œë°”ë…¸??ì¶”ì¶œ ?ŒìŠ¤??- /test_clova <URL>"""
        if not context.args:
            await update.message.reply_text("?¬ìš©ë²? /test_clova <?´ë¡œë°”ë…¸?¸URL>")
            return

        url = context.args[0]
        await update.message.reply_text(f"?” ?ŒìŠ¤???œì‘: {url}")

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, fetch_clova_transcript, url)
            if result and len(result) > 50:
                preview = result[:500] + ("..." if len(result) > 500 else "")
                await update.message.reply_text(
                    f"??ì¶”ì¶œ ?±ê³µ!\nê¸¸ì´: {len(result)}??n\në¯¸ë¦¬ë³´ê¸°:\n{preview}")
            else:
                await update.message.reply_text(
                    f"??ì¶”ì¶œ ?¤íŒ¨\nê²°ê³¼ ê¸¸ì´: {len(result) if result else 0}??n"
                    f"?´ìš©: {result[:200] if result else '(?†ìŒ)'}")
        except Exception as e:
            await update.message.reply_text(f"???ëŸ¬: {str(e)[:500]}")

    async def login_naver_cmd(update: Update, context):
        """?¤ì´ë²?ë¡œê·¸????ì¿ í‚¤ ?€??- /login_naver"""
        await update.message.reply_text(
            "?Œ ?¤ì´ë²?ë¡œê·¸??ì°½ì„ ?½ë‹ˆ??..\n"
            "PC??Chrome ì°½ì´ ?¨ë©´ ë¡œê·¸?¸í•´ì£¼ì„¸??\n"
            "??ìµœë? 2ë¶??€ê¸°í•©?ˆë‹¤.")

        loop = asyncio.get_event_loop()
        try:
            success = await loop.run_in_executor(None, naver_login_interactive)
            if success:
                cookies = _load_naver_cookies()
                await update.message.reply_text(
                    f"???¤ì´ë²?ë¡œê·¸???„ë£Œ!\n?ª ì¿ í‚¤ {len(cookies)}ê°??€?¥ë¨\n\n"
                    f"?´ì œ ?´ë¡œë°”ë…¸??URL??ë³´ë‚´ë©??ë™?¼ë¡œ ?ìŠ¤?¸ë? ì¶”ì¶œ?©ë‹ˆ??")
            else:
                await update.message.reply_text("??ë¡œê·¸???¤íŒ¨. ?¤ì‹œ ?œë„?´ì£¼?¸ìš”.")
        except Exception as e:
            await update.message.reply_text(f"???ëŸ¬: {str(e)[:300]}")

    # ì»¨ì½œ ??    concall_queue = _queue.Queue()

    async def concall_worker(bot):
        """ì»¨ì½œ ???Œì»¤"""
        while True:
            if not concall_queue.empty():
                item = concall_queue.get()
                remaining = concall_queue.qsize()
                try:
                    await bot.edit_message_text(
                        chat_id=item["chat_id"], message_id=item["msg_id"],
                        text=f"?™ï¸?*{item['company']}*\n?“ ?ìŠ¤??ì¶”ì¶œ ì¤?..{f' (?€ê¸?{remaining}ê±?' if remaining else ''}")

                    loop = asyncio.get_event_loop()
                    transcript = await loop.run_in_executor(None, fetch_clova_transcript, item["url"])

                    if not transcript or len(transcript) < 50:
                        await bot.edit_message_text(
                            chat_id=item["chat_id"], message_id=item["msg_id"],
                            text=f"? ï¸ *{item['company']}*\n?ìŠ¤??ì¶”ì¶œ ?¤íŒ¨.\n?’¡ /login_naver ë¡??¤ì´ë²?ë¡œê·¸?????¤ì‹œ ?œë„?´ì£¼?¸ìš”.")
                        if not concall_queue.empty():
                            await asyncio.sleep(5)
                        continue

                    await bot.edit_message_text(
                        chat_id=item["chat_id"], message_id=item["msg_id"],
                        text=f"?™ï¸?*{item['company']}*\n?¤– AI ?”ì•½ ì¤?.. ({len(transcript)}??")

                    parsed = await loop.run_in_executor(None, parse_concall, transcript, item["company"], item["host"])
                    await loop.run_in_executor(None, save_concall, parsed, item["url"],
                                              item["time"], item["date"], item["host"], transcript)

                    # ê°ì • ?´ëª¨ì§€
                    sent_emoji = {"positive": "?Ÿ¢", "negative": "?”´"}.get(parsed.get("sentiment", ""), "?Ÿ¡")
                    kp = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(parsed.get("key_points", [])[:5]))
                    qa = "\n".join(f"  ?’¬ {q}" for q in (parsed.get("qa_highlights", [])[:3]))

                    dash_url = get_dashboard_url().replace("/dashboard", "/concalls")
                    text = f"?™ï¸?*{parsed.get('company_name', item['company'])}* ì»¨ì½œ ?”ì•½\n"
                    text += f"?â”?â”?â”?â”?â”?â”?â”?â”\n"
                    if item["host"]: text += f"?¦ {item['host']} | "
                    if item["time"]: text += f"??{item['time']}\n"
                    text += f"{sent_emoji} ?? {parsed.get('sentiment', 'neutral')}\n"

                    if parsed.get("headline"):
                        text += f"\n?’¡ *{parsed['headline']}*\n"
                    if kp:
                        text += f"\n?“‹ *?µì‹¬ ?¬ì¸??\n{kp}\n"
                    if parsed.get("guidance"):
                        text += f"\n?¯ *ê°€?´ë˜??\n  {parsed['guidance']}\n"
                    if qa:
                        text += f"\n??*Q&A ?˜ì´?¼ì´??\n{qa}\n"
                    text += f"\n?”— [ì»¨ì½œ ?€?œë³´??({dash_url})"

                    await bot.edit_message_text(
                        chat_id=item["chat_id"], message_id=item["msg_id"],
                        text=text, disable_web_page_preview=True)
                except Exception as e:
                    await bot.edit_message_text(
                        chat_id=item["chat_id"], message_id=item["msg_id"],
                        text=f"??*{item['company']}* ì»¨ì½œ ì²˜ë¦¬ ?¤ë¥˜\n{str(e)[:200]}")

                if not concall_queue.empty():
                    await asyncio.sleep(30)
            else:
                await asyncio.sleep(1)

    async def handle_text(update: Update, context):
        """?ìŠ¤??ë©”ì‹œì§€ ì²˜ë¦¬ - ?´ë¡œë°”ë…¸??URL ê°ì?"""
        save_chat_id(update.message.chat_id)
        text = update.message.text or ""

        # ?´ë¡œë°”ë…¸??URL ì°¾ê¸°
        urls = re.findall(r'https?://clovanote\.naver\.com/s/\S+', text)
        if not urls:
            return

        # ?ìŠ¤?¸ì—??ì»¨ì½œ ?•ë³´ ?Œì‹±
        lines = text.strip().split("\n")
        today_str = datetime.now().strftime("%Y-%m-%d")

        for url in urls:
            # URL ì§ì „ ì¤„ì—???•ë³´ ì¶”ì¶œ
            company = ""
            host = ""
            ctime = ""

            for i, line in enumerate(lines):
                if url in line or (i + 1 < len(lines) and url in lines[i + 1]):
                    # "14:00 ?¬ìŠ¤ì½”ì¸?°ë‚´?”ë„ ?€?? ?¨í„´
                    m = re.match(r'(\d{1,2}:\d{2})\s+(.+?)(?:\s+([\wê°€-??+))?\s*$', line.strip())
                    if m:
                        ctime = m.group(1)
                        rest = m.group(2).strip()
                        # ë§ˆì?ë§??¨ì–´ê°€ ì¦ê¶Œ?¬ì¸ì§€ ?•ì¸ (2~6ê¸€???œêµ­??
                        parts = rest.rsplit(None, 1)
                        if len(parts) == 2 and re.match(r'^[ê°€-?£A-Z]{1,10}$', parts[1]):
                            company = parts[0].strip()
                            host = parts[1].strip()
                        else:
                            company = rest
                    elif not url in line:
                        company = line.strip()
                    break

            if not company:
                company = "ì»¨ì½œ"

            # ?ì— ì¶”ê?
            waiting = concall_queue.qsize()
            if waiting > 0:
                msg = await update.message.reply_text(
                    f"?™ï¸?*{company}*\n???€ê¸?ì¤?({waiting}ê±??ì— ì²˜ë¦¬ ì¤?")
            else:
                msg = await update.message.reply_text(
                    f"?™ï¸?*{company}*\n?“ ì²˜ë¦¬ ?œì‘...")

            concall_queue.put({
                "chat_id": update.message.chat_id,
                "msg_id": msg.message_id,
                "url": url.strip(),
                "company": company,
                "host": host,
                "time": ctime,
                "date": today_str,
            })

    def bot_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tg_app = Application.builder().token(TELEGRAM_TOKEN).build()
        tg_app.add_handler(CommandHandler("start", start_cmd))
        tg_app.add_handler(CommandHandler("today", today_cmd))
        tg_app.add_handler(CommandHandler("stats", stats_cmd))
        tg_app.add_handler(CommandHandler("test_clova", test_clova_cmd))
        tg_app.add_handler(CommandHandler("login_naver", login_naver_cmd))
        tg_app.add_handler(MessageHandler(filters.Document.ALL, handle_doc))
        tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

        # ???Œì»¤ ?±ë¡
        async def post_init(app):
            asyncio.create_task(queue_worker(app.bot))
            asyncio.create_task(concall_worker(app.bot))
        tg_app.post_init = post_init

        logger.info("?¤– ?”ë ˆê·¸ë¨ ë´??œì‘")
        tg_app.run_polling(allowed_updates=Update.ALL_TYPES)

    threading.Thread(target=bot_thread, daemon=True).start()


# ?€?€ ?¼ì¼ ë¸Œë¦¬???€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
def generate_briefing(date_str):
    """? ì§œë³?ë¸Œë¦¬???ìŠ¤???ì„±"""
    conn = get_db()
    reports = conn.execute(
        "SELECT * FROM reports WHERE report_date = ? ORDER BY report_type, created_at",
        [date_str]
    ).fetchall()
    upgrades = conn.execute(
        "SELECT * FROM reports WHERE report_date = ? AND prev_rating != ''",
        [date_str]
    ).fetchall()
    concalls = conn.execute(
        "SELECT * FROM concalls WHERE concall_date = ? ORDER BY concall_time, created_at",
        [date_str]
    ).fetchall()
    conn.close()

    if not reports and not concalls:
        return f"?“­ *{date_str}*\në¦¬í¬?¸ê? ?†ìŠµ?ˆë‹¤."

    dash_url = get_dashboard_url()
    weekdays_kr = ["??, "??, "??, "ëª?, "ê¸?, "??, "??]
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        wd = weekdays_kr[dt.weekday()]
        date_display = f"{dt.month}/{dt.day}({wd})"
    except:
        date_display = date_str

    text = f"?Œ… *{date_display} ëª¨ë‹ ë¸Œë¦¬??\n"
    text += f"?â”?â”?â”?â”?â”?â”?â”?â”\n"
    text += f"?“Š ?„ì²´ {len(reports)}ê±?

    stocks = [r for r in reports if r["report_type"] == "stock"]
    industries = [r for r in reports if r["report_type"] == "industry"]
    strategies = [r for r in reports if r["report_type"] == "strategy"]

    if stocks: text += f" Â· ì¢…ëª© {len(stocks)}"
    if industries: text += f" Â· ?°ì—… {len(industries)}"
    if strategies: text += f" Â· ?„ëµ {len(strategies)}"
    text += "\n"

    if upgrades:
        text += f"\n?”„ *?¬ì?˜ê²¬ ë³€ê²?({len(upgrades)}ê±?*\n"
        for r in upgrades[:5]:
            emoji = "?Ÿ¢" if r["rating"] in ["Buy", "Overweight", "Outperform"] else "?”´" if r["rating"] in ["Sell", "Underweight"] else "?Ÿ¡"
            text += f"  {emoji} {r['stock_name']} ({r['brokerage']})\n"
            text += f"     {r['prev_rating']} ??*{r['rating']}*"
            if r["prev_target_price"] and r["target_price"]:
                text += f" | TP {r['prev_target_price']}??r['target_price']}"
            text += "\n"

    if strategies:
        text += f"\n?Œ *?œì¥ ?„ëµ*\n"
        for r in strategies[:3]:
            text += f"  Â· {r['brokerage']}: {r['headline'] or r['report_title']}\n"

    if industries:
        text += f"\n?­ *?°ì—… ë¶„ì„*\n"
        for r in industries[:3]:
            text += f"  Â· {r['brokerage']}: {r['headline'] or r['report_title']}\n"

    if stocks:
        text += f"\n?“Š *ì¢…ëª© ë¦¬í¬??\n"
        for r in stocks[:8]:
            emoji = {"Buy": "?Ÿ¢", "Overweight": "?Ÿ¢", "Outperform": "?Ÿ¢",
                    "Neutral": "?Ÿ¡", "Hold": "?Ÿ¡", "Sell": "?”´", "Underweight": "?”´"}.get(r["rating"], "??)
            tp = f" TP {r['target_price']}" if r["target_price"] else ""
            text += f"  {emoji} *{r['stock_name']}* {r['brokerage']} {r['rating']}{tp}\n"

    if concalls:
        text += f"\n?™ï¸?*ì»¨í¼?°ìŠ¤ì½?({len(concalls)}ê±?*\n"
        for c in concalls[:5]:
            sent = c["sentiment"] if "sentiment" in c.keys() else ""
            se = {"positive": "?Ÿ¢", "negative": "?”´"}.get(sent, "?Ÿ¡")
            cn = c["company_name"] if "company_name" in c.keys() else ""
            text += f"  {se} {cn}"
            ho = c["host"] if "host" in c.keys() else ""
            if ho:
                text += f" ({ho})"
            text += "\n"
            hl = c["headline"] if "headline" in c.keys() else ""
            if hl:
                text += f"    {hl}\n"

    text += f"\n?”— [?€?œë³´?œì—???ì„¸ ë³´ê¸°]({dash_url}?date={date_str})"
    return text
def run_daily_briefing():
    """ëª¨ë‹(7:30) + ?ì‹¬(13:00) ë¸Œë¦¬???ë™ ë°œì†¡"""
    if not TELEGRAM_TOKEN:
        return

    def send_briefing(brief_type="morning"):
        """ë¸Œë¦¬???ì„± ë°?ë°œì†¡"""
        now = datetime.now()
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        today = now.strftime("%Y-%m-%d")

        conn = get_db()
        y_cnt = conn.execute("SELECT COUNT(*) FROM reports WHERE report_date = ?", [yesterday]).fetchone()[0]
        t_cnt = conn.execute("SELECT COUNT(*) FROM reports WHERE report_date = ?", [today]).fetchone()[0]
        conn.close()

        brief_date = today if t_cnt > 0 else yesterday
        if y_cnt == 0 and t_cnt == 0:
            return

        briefing = generate_briefing(brief_date)
        # ?ì‹¬?€ ?¤ë” ë³€ê²?        if brief_type == "lunch":
            briefing = briefing.replace("?Œ…", "?€ï¸?).replace("ëª¨ë‹ ë¸Œë¦¬??, "?ì‹¬ ë¸Œë¦¬??)

        try:
            chat_ids = []
            cid_file = BASE_DIR / "chat_ids.txt"
            if cid_file.exists():
                chat_ids = [l.strip() for l in cid_file.read_text().splitlines() if l.strip()]
            for cid in chat_ids:
                try:
                    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
                    httpx.post(url, json={"chat_id": cid, "text": briefing,
                              "parse_mode": "Markdown", "disable_web_page_preview": True}, timeout=10)
                except:
                    pass
            logger.info(f"?“¬ {brief_type} ë¸Œë¦¬??ë°œì†¡ ?„ë£Œ ({len(chat_ids)}ëª?")
        except Exception as e:
            logger.error(f"ë¸Œë¦¬??ë°œì†¡ ?¤íŒ¨: {e}")

    def briefing_loop():
        sent_morning = False
        sent_lunch = False
        while True:
            now = datetime.now()

            # ëª¨ë‹ ë¸Œë¦¬??07:30
            if now.hour == 7 and now.minute == 30 and not sent_morning:
                send_briefing("morning")
                sent_morning = True

            # ?ì‹¬ ë¸Œë¦¬??13:00
            if now.hour == 13 and now.minute == 0 and not sent_lunch:
                send_briefing("lunch")
                sent_lunch = True

            # ?ì • ë¦¬ì…‹
            if now.hour == 0 and now.minute == 0:
                sent_morning = False
                sent_lunch = False

            time.sleep(30)

    threading.Thread(target=briefing_loop, daemon=True).start()


def save_chat_id(chat_id):
    """ë´??¬ìš©??chat_id ?€??""
    cid_file = BASE_DIR / "chat_ids.txt"
    existing = set()
    if cid_file.exists():
        existing = set(l.strip() for l in cid_file.read_text().splitlines() if l.strip())
    cid = str(chat_id)
    if cid not in existing:
        with open(cid_file, "a") as f:
            f.write(f"{cid}\n")
        logger.info(f"?“ ??chat_id ?€?? {cid}")


# ?€?€ ë©”ì¸ ?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€?€
NGROK_URL = ""


def start_ngrok():
    """ngrok ?°ë„ - ?¸ë? ?‘ì† URL ?ì„±"""
    global NGROK_URL
    try:
        import subprocess
        subprocess.run(["taskkill", "/f", "/im", "ngrok.exe"], capture_output=True)
        time.sleep(1)
        subprocess.Popen(["ngrok", "http", "5000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        resp = httpx.get("http://127.0.0.1:4040/api/tunnels", timeout=5)
        for t in resp.json().get("tunnels", []):
            if t.get("proto") == "https":
                NGROK_URL = t["public_url"]
                (BASE_DIR / "ngrok_url.txt").write_text(NGROK_URL, encoding="utf-8")
                return NGROK_URL
    except:
        pass
    return None


@app.route("/share")
@login_required
def share_url():
    """?¸ë? ê³µìœ  URL ?•ì¸ ?˜ì´ì§€"""
    global NGROK_URL
    try:
        resp = httpx.get("http://127.0.0.1:4040/api/tunnels", timeout=3)
        for t in resp.json().get("tunnels", []):
            if t.get("proto") == "https":
                NGROK_URL = t["public_url"]
    except:
        pass

    if NGROK_URL:
        url_html = (
            "<div class='url' "
            "onclick='navigator.clipboard.writeText(this.textContent).then(()=>alert(\'ë³µì‚¬??\'))'>"
            f"{NGROK_URL}/dashboard"
            "</div><p>?ï¸ ?´ë¦­?˜ë©´ ë³µì‚¬</p>"
        )
    else:
        url_html = "<p style='color:#ef4444'>ngrok ë¯¸ì‹¤??/p>"

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>body{{font-family:'Noto Sans KR',sans-serif;background:#0a0e17;color:#e0e6f0;display:flex;
flex-direction:column;align-items:center;justify-content:center;min-height:100vh;text-align:center}}
.url{{font-size:18px;color:#3b82f6;background:#111827;padding:16px 28px;border-radius:12px;
border:1px solid #1e2a42;margin:16px;word-break:break-all;cursor:pointer}}
.url:hover{{background:#1a2540}}a{{color:#4a5a7a;font-size:13px}}</style></head>
<body><h1>?Œ ?¸ë? ê³µìœ  ì£¼ì†Œ</h1>
{url_html}
<a href="/dashboard">???€?œë³´??/a></body></html>"""
if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("??.env??GROQ_API_KEY ?„ìš”"); sys.exit(1)
    init_db()
    print("=" * 50)
    print("?“Š ì¦ê¶Œ??ë¦¬í¬???€?œë³´??v3")
    print("=" * 50)
    print(f"?Œ ë¡œì»¬: http://localhost:5000/dashboard")
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"?  ?´ë?ë§? http://{local_ip}:5000/dashboard")
    except:
        pass
    ngrok_url = start_ngrok()
    if ngrok_url:
        print(f"?Œ ?¸ë??‘ì†: {ngrok_url}/dashboard")
    else:
        print("?’¡ ?¸ë??‘ì†: ngrok ?¤ì¹˜ ???ë™ ?°ê²° (? íƒ?¬í•­)")
    print(f"?“¤ ?…ë¡œ?? http://localhost:5000/upload")
    print(f"?™ï¸?ì»¨ì½œ: http://localhost:5000/concalls")
    print(f"?“… ?„í´ë¦? http://localhost:5000/weekly")
    print(f"?“ˆ ì»¨ì„¼?œìŠ¤: http://localhost:5000/consensus")
    print(f"?”— ê³µìœ URL: http://localhost:5000/share")
    if TELEGRAM_TOKEN: print("?¤– ?”ë ˆê·¸ë¨ ë´? ON")
    if NOTION_TOKEN: print("?—„ï¸?Notion: ON")
    print("=" * 50)
    run_telegram_bot()
    run_daily_briefing()
    app.run(host="0.0.0.0", port=5000, debug=False)
