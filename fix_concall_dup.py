import re

code = open('app.py', 'r', encoding='utf-8').read()

old = '''    conn = get_db()
    kp = json.dumps(data.get("key_points", []), ensure_ascii=False)'''

new = '''    conn = get_db()
    # 컨콜 중복 체크 (같은 clova_url이면 스킵)
    if clova_url:
        existing = conn.execute("SELECT id FROM concalls WHERE clova_url = ?", [clova_url]).fetchone()
        if existing:
            logger.info(f"컨콜 중복 스킵: {clova_url}")
            conn.close()
            return existing[0]
    kp = json.dumps(data.get("key_points", []), ensure_ascii=False)'''

if old in code:
    code = code.replace(old, new, 1)
    open('app.py', 'w', encoding='utf-8').write(code)
    print('done - concall duplicate check added')
else:
    print('ERROR: pattern not found')
