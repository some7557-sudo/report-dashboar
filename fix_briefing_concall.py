code = open('app.py', 'r', encoding='utf-8').read()
changes = 0

# 1. concalls 조회 추가
if 'SELECT * FROM concalls WHERE concall_date' not in code:
    old1 = "prev_rating != ''\", [date_str]).fetchall()\n    conn.close()"
    new1 = "prev_rating != ''\", [date_str]).fetchall()\n    concalls = conn.execute(\"SELECT * FROM concalls WHERE concall_date = ?\", [date_str]).fetchall()\n    conn.close()"
    if old1 in code:
        code = code.replace(old1, new1, 1)
        changes += 1
        print('1. concalls query added')
    else:
        print('1. ERROR - query pattern not found')

# 2. 리포트+컨콜 둘다 없을때만 리턴
if 'not concalls' not in code:
    old2 = 'if not reports:\n        return f"'
    new2 = 'if not reports and not concalls:\n        return f"'
    if old2 in code:
        code = code.replace(old2, new2, 1)
        changes += 1
        print('2. empty check updated')

# 3. 컨콜 섹션 추가 (dash_url?date= 링크 앞에)
if 'concalls[:5]' not in code:
    target = 'dash_url}?date={date_str})"'
    idx = code.find(target)
    if idx > 0:
        line_start = code.rfind('\n', 0, idx) + 1
        
        concall_section = (
            '    # \ucee8\ucf5c \uc139\uc158\n'
            '    if concalls:\n'
            '        text += f"\\n\U0001f399\ufe0f *\ucee8\ud37c\ub7f0\uc2a4\ucf5c ({len(concalls)}\uac74)*\\n"\n'
            '        for c in concalls[:5]:\n'
            '            sent = c["sentiment"] if "sentiment" in c.keys() else ""\n'
            '            se = {"positive": "\U0001f7e2", "negative": "\U0001f534"}.get(sent, "\U0001f7e1")\n'
            '            cn = c["company_name"] if "company_name" in c.keys() else ""\n'
            '            text += f"  {se} {cn}"\n'
            '            ho = c["host"] if "host" in c.keys() else ""\n'
            '            if ho:\n'
            '                text += f" ({ho})"\n'
            '            text += "\\n"\n'
            '            hl = c["headline"] if "headline" in c.keys() else ""\n'
            '            if hl:\n'
            '                text += f"    {hl}\\n"\n'
            '\n'
        )
        
        code = code[:line_start] + concall_section + code[line_start:]
        changes += 1
        print('3. concall section added')
    else:
        print('3. ERROR - link not found')

if changes > 0:
    open('app.py', 'w', encoding='utf-8').write(code)
    print(f'\ndone - {changes} changes')
else:
    print('no changes')
