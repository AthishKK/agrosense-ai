with open('app.py', encoding='utf-8') as f:
    c = f.read()

old = "result[current_section] = '\n'.join(current_lines).strip()"
new = "result[current_section] = chr(10).join(current_lines).strip()"

count = c.count(old)
print(f"Found {count} occurrence(s)")

if count == 0:
    print("NO MATCH")
else:
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(c.replace(old, new))
    print("Done")
