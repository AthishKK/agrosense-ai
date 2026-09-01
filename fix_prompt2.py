with open('utils/ai_advice.py', encoding='utf-8') as f:
    c = f.read()

old_prompt = """    prompt = f\"\"\"Provide farming advice for {crop} crop. Write exactly 6 sections IN THIS EXACT ORDER. Number each section heading as shown below. Use simple language a village farmer can understand.

Farmer data: Crop={crop}, Season={season}, Soil={soil_type}, N={N} kg/ha, P={P} kg/ha, K={K} kg/ha, pH={ph}, Temp={temperature}\u00b0C, Humidity={humidity}%, Rainfall={rainfall}mm, Fertilizer={fertilizer}, Yield={yield_range} t/ha.

1. \U0001f33e SEED SELECTION
[Write 2-3 best varieties for {crop} in {season} on {soil_type} soil. Include seed rate as a range in kg/acre, seed treatment, sowing depth and spacing in cm.]

2. \U0001f48a FERTILIZER SCHEDULE
[Write week by week fertilizer plan. All quantities must be ranges like 10-15 kg/acre. Include micronutrient tips for pH {ph} and one foliar spray with dilution ratio.]

3. \U0001f4a7 IRRIGATION PLAN
[Use simple farmer-friendly language. Write weeks after planting instead of DAS. Use simple water quantity descriptions instead of mm. Write days after sowing for first irrigation. Frequency and quantity per growth stage. Critical stress stages. Water saving tip for {rainfall}mm annual rainfall.]

4. \U0001f9a0 DISEASE & PEST PREVENTION
[Disease risk note: {disease_risk_note}. Write top 2 pests for {crop} at {temperature}\u00b0C with preventive spray dose ranges. Include one organic alternative.]

5. \u26a0\ufe0f EXPECTED CHALLENGES
[Write exactly 3 challenges based on the given conditions. For each: challenge name, how to detect it, and how to mitigate it.]

6. \U0001f33f HARVEST GUIDANCE
[Write days after sowing to harvest for {crop}. List 4 visual maturity signs. Describe post-harvest drying and storage method and duration.]

End your response with this exact line:
\u2139\ufe0f These recommendations are based on verified agronomic guidelines from ICAR and FAO integrated into AgroSense AI. Field conditions vary \u2014 monitor your crop regularly and adjust as needed.\"\"\""""

new_prompt = """    prompt = f\"\"\"Write farming advice for {crop} farmers.
Use simple village farmer language. No technical terms.

SECTION 1 - SEED SELECTION:
Write 2-3 best {crop} varieties for {season} in {soil_type} soil.
Seed quantity needed per acre as a range.
How to treat seeds before planting.
How deep to plant and spacing between plants.

SECTION 2 - FERTILIZER SCHEDULE:
Week by week fertilizer plan from planting to harvest.
Write fertilizer amounts as ranges in kg per acre only.
Which fertilizers to use for pH {ph} soil.
One foliar spray tip with mixing ratio.

SECTION 3 - IRRIGATION PLAN:
When to water first after planting (write in days or weeks).
How often and how much water at each growth stage.
Most important time not to miss watering.
How to save water with only {rainfall}mm yearly rainfall.

SECTION 4 - DISEASE AND PEST PREVENTION:
Disease situation: {disease_risk_note}
Top 2 pests that attack {crop} in {temperature}\u00b0C weather.
Preventive spray for each pest with dose range.
One natural/organic option.

SECTION 5 - EXPECTED CHALLENGES:
Write exactly 3 problems farmers face growing {crop}
in these conditions: {temperature}\u00b0C temperature,
{humidity}% humidity, {rainfall}mm rainfall, pH {ph}.
For each: what is the problem, how to spot it, how to fix it.

SECTION 6 - HARVEST GUIDANCE:
How many days after planting to harvest {crop}.
4 signs that {crop} is ready to harvest.
How to dry after harvest and for how long.
How to store and for how long.

End with this exact line:
\u2139\ufe0f These recommendations are based on verified agronomic guidelines from ICAR and FAO integrated into AgroSense AI. Field conditions vary \u2014 monitor your crop regularly and adjust as needed.\"\"\""""

old_system = "'You are a concise agricultural advisor. Write only the 6 requested farming advice sections with their emoji headings. No planning, no notes, no thinking process \u2014 just direct advice.'"
new_system = "'You are a helpful farming advisor for Indian village farmers. Write clear simple advice using the exact 6 sections provided. Never repeat any section. Never show your thinking. Output only the farming advice.'"

for old, new, label in [
    (old_prompt, new_prompt, "prompt"),
    (old_system, new_system, "system message"),
]:
    if old not in c:
        print(f"NO MATCH: {label}")
    else:
        c = c.replace(old, new, 1)
        print(f"OK: {label}")

with open('utils/ai_advice.py', 'w', encoding='utf-8') as f:
    f.write(c)
print("Done")
