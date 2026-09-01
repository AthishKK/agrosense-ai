with open('utils/ai_advice.py', encoding='utf-8') as f:
    c = f.read()

replacements = [
    (
        "prompt = f\"\"\"Provide farming advice for {crop} crop. Write exactly 6 sections using these headings. Start each section directly with the emoji heading on its own line, then the content.",
        "prompt = f\"\"\"Provide farming advice for {crop} crop. Write exactly 6 sections IN THIS EXACT ORDER. Number each section heading as shown below. Use simple language a village farmer can understand."
    ),
    (
        "\U0001f33e SEED SELECTION",
        "1. \U0001f33e SEED SELECTION"
    ),
    (
        "\U0001f48a FERTILIZER SCHEDULE",
        "2. \U0001f48a FERTILIZER SCHEDULE"
    ),
    (
        "\U0001f4a7 IRRIGATION PLAN",
        "3. \U0001f4a7 IRRIGATION PLAN"
    ),
    (
        "\U0001f9a0 DISEASE & PEST PREVENTION",
        "4. \U0001f9a0 DISEASE & PEST PREVENTION"
    ),
    (
        "\u26a0\ufe0f EXPECTED CHALLENGES",
        "5. \u26a0\ufe0f EXPECTED CHALLENGES"
    ),
    (
        "\U0001f33f HARVEST GUIDANCE",
        "6. \U0001f33f HARVEST GUIDANCE"
    ),
    (
        "[Write days after sowing for first irrigation.",
        "[Use simple farmer-friendly language. Write weeks after planting instead of DAS. Use simple water quantity descriptions instead of mm. Write days after sowing for first irrigation."
    ),
]

matched = 0
for old, new in replacements:
    if old in c:
        c = c.replace(old, new, 1)
        matched += 1
    else:
        print(f"NO MATCH #{matched+1}")

with open('utils/ai_advice.py', 'w', encoding='utf-8') as f:
    f.write(c)
print(f"Done — {matched}/{len(replacements)} replacements applied")
