with open('app.py', encoding='utf-8') as f:
    c = f.read()

old = "for line in advice_text.split('\n'):"
new = "for line in advice_text.splitlines():"

if old not in c:
    print("NO MATCH")
else:
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(c.replace(old, new, 1))
    print("Done")
