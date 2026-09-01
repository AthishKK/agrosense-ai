import re

with open('app.py', 'r', encoding='utf-8') as f:
    raw = f.read()

# Match the entire parse_advice_sections function
pattern = r'(def parse_advice_sections\(advice_text\):).*?(return result, section_map, display_names)'
replacement = '''def parse_advice_sections(advice_text):

    section_map = {
        "SEED SELECTION":      "\U0001f33e",
        "SEED":                "\U0001f33e",
        "FERTILIZER SCHEDULE": "\U0001f48a",
        "FERTILIZER":          "\U0001f48a",
        "IRRIGATION PLAN":     "\U0001f4a7",
        "IRRIGATION":          "\U0001f4a7",
        "DISEASE & PEST":      "\U0001f9a0",
        "DISEASE AND PEST":    "\U0001f9a0",
        "DISEASE":             "\U0001f9a0",
        "PEST":                "\U0001f9a0",
        "EXPECTED CHALLENGES": "\u26a0\ufe0f",
        "CHALLENGES":          "\u26a0\ufe0f",
        "HARVEST GUIDANCE":    "\U0001f33f",
        "HARVEST":             "\U0001f33f",
    }

    display_names = {
        "SEED SELECTION":      "Seed Selection",
        "SEED":                "Seed Selection",
        "FERTILIZER SCHEDULE": "Fertilizer Schedule",
        "FERTILIZER":          "Fertilizer Schedule",
        "IRRIGATION PLAN":     "Irrigation Plan",
        "IRRIGATION":          "Irrigation Plan",
        "DISEASE & PEST":      "Disease & Pest Prevention",
        "DISEASE AND PEST":    "Disease & Pest Prevention",
        "DISEASE":             "Disease & Pest Prevention",
        "PEST":                "Disease & Pest Prevention",
        "EXPECTED CHALLENGES": "Expected Challenges",
        "CHALLENGES":          "Expected Challenges",
        "HARVEST GUIDANCE":    "Harvest Guidance",
        "HARVEST":             "Harvest Guidance",
    }

    result = {}
    current_section = None
    current_lines = []

    for line in advice_text.split(\'\\n\'):
        line_upper = line.strip().upper()
        line_upper = line_upper.replace(\'#\',\'\').replace(\'*\',\'\').replace(\'-\',\'\').replace(\'\U0001f33e\',\'\').replace(\'\U0001f48a\',\'\').replace(\'\U0001f4a7\',\'\').replace(\'\U0001f9a0\',\'\').replace(\'\u26a0\ufe0f\',\'\').replace(\'\U0001f33f\',\'\').strip()

        matched = False
        matched_key = None

        for section in section_map:
            if section in line_upper and len(line_upper) < 50:
                matched_key = section
                matched = True
                break

        if matched:
            if current_section and current_lines:
                result[current_section] = \'\\n\'.join(current_lines).strip()
            current_section = matched_key
            current_lines = []
        elif current_section:
            current_lines.append(line)

    if current_section and current_lines:
        result[current_section] = \'\\n\'.join(current_lines).strip()

    result = {k: v for k, v in result.items() if v.strip()}

    return result, section_map, display_names'''

new_raw = re.sub(pattern, replacement, raw, flags=re.DOTALL)

if new_raw == raw:
    print("NO MATCH")
else:
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(new_raw)
    print("Done")
