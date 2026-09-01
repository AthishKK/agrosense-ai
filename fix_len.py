with open('app.py', encoding='utf-8') as f:
    c = f.read()

old = "if section in line_upper and len(line_upper) < 50:"
new = "if section in line_upper and len(line_upper) < 80:"

if old not in c:
    print("NO MATCH")
else:
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(c.replace(old, new, 1))
    print("Done")
