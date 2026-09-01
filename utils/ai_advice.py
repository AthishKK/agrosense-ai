import sys
sys.stdout.reconfigure(encoding='utf-8')
import re

from groq import Groq


def configure_groq(api_key):
    return Groq(api_key=api_key)


def get_farming_advice(client, crop, fertilizer, yield_range,
                       N, P, K, ph, temperature, humidity,
                       rainfall, soil_type, season):

    if temperature > 28 and humidity > 80:
        disease_risk_note = (
            f"Current conditions (temp {temperature}°C, humidity {humidity}%) "
            "elevate fungal disease risk if they persist."
        )
    elif temperature < 20:
        disease_risk_note = (
            f"Current temperature ({temperature}°C) elevates rust and mildew risk "
            "if conditions persist."
        )
    else:
        disease_risk_note = (
            f"Current conditions (temp {temperature}°C, humidity {humidity}%) "
            "are currently safe — monitor regularly."
        )

    prompt = (
        f"Write detailed farming advice for {crop} crop.\n"
        f"Season: {season}, Soil: {soil_type}\n"
        f"N={N} kg/ha, P={P} kg/ha, K={K} kg/ha, pH={ph}\n"
        f"Temperature: {temperature}°C, Humidity: {humidity}%, "
        f"Rainfall: {rainfall}mm\n"
        f"Recommended fertilizer: {fertilizer}\n"
        f"Expected yield: {yield_range} t/ha\n\n"
        f"Write ALL 6 sections below with bullet points. "
        f"Each section must have at least 5 bullet points. "
        f"Complete each section fully before starting the next.\n\n"
        f"## Seed Selection\n"
        f"- Best varieties of {crop} for {season} in {soil_type} soil (list 2-3)\n"
        f"- Seed rate range in kg/acre\n"
        f"- Seed treatment: product name and dose\n"
        f"- Sowing depth in cm\n"
        f"- Plant spacing: row x plant in cm\n"
        f"- Germination tip for {temperature}°C\n\n"
        f"## Fertilizer Schedule\n"
        f"- Week 0 (Sowing): Urea, DAP, MOP doses as ranges in kg/acre\n"
        f"- Week 3-4: doses as ranges in kg/acre\n"
        f"- Week 6-8: doses as ranges in kg/acre\n"
        f"- Week 10-12: doses as ranges in kg/acre\n"
        f"- Micronutrient for pH {ph}: product and dose range\n"
        f"- Foliar spray: product, concentration %, dilution ratio\n\n"
        f"## Irrigation Plan\n"
        f"- First irrigation: exact days after sowing\n"
        f"- Seedling stage: frequency and mm per irrigation\n"
        f"- Vegetative stage: frequency and mm per irrigation\n"
        f"- Flowering stage: frequency and mm per irrigation\n"
        f"- Most critical stage for {crop}: name it\n"
        f"- Water saving tip for {rainfall}mm annual rainfall\n\n"
        f"## Disease & Pest Prevention\n"
        f"- Disease status: {disease_risk_note}\n"
        f"- Pest 1: name, scientific name, symptom, spray product and dose range\n"
        f"- Pest 2: name, scientific name, symptom, spray product and dose range\n"
        f"- Organic alternative: product, concentration, frequency\n\n"
        f"## Expected Challenges\n"
        f"- Challenge 1: problem name, detection signs, specific solution\n"
        f"- Challenge 2: problem name, detection signs, specific solution\n"
        f"- Challenge 3: problem name, detection signs, specific solution\n\n"
        f"## Harvest Guidance\n"
        f"- Days after sowing to harvest {crop}\n"
        f"- Maturity sign 1\n"
        f"- Maturity sign 2\n"
        f"- Maturity sign 3\n"
        f"- Maturity sign 4\n"
        f"- Drying method and duration\n"
        f"- Storage: container, temperature, humidity, duration\n\n"
        f"End with this exact line:\n"
        f"These recommendations are based on verified agronomic guidelines "
        f"from ICAR and FAO integrated into AgroSense AI. "
        f"Field conditions vary — monitor your crop regularly and adjust as needed."
    )

    response = client.chat.completions.create(
        model='openai/gpt-oss-20b',
        messages=[
            {
                'role': 'system',
                'content': (
                    'You are a senior agronomist with 20 years experience. '
                    'Write detailed technical farming advice. '
                    'Use specific product names and exact ranges for every measurement. '
                    'Complete all bullet points under each section before moving to the next section. '
                    'Never leave any section empty. Never repeat a section heading.'
                )
            },
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=2500,
        temperature=0.3
    )

    content = response.choices[0].message.content

    if '<think>' in content:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    return content.strip()


def get_comparison_advice(client, crop1, crop2, confidence1,
                          confidence2, season, soil_type):

    prompt = (
        f"Compare {crop1} ({confidence1:.1f}% confidence) vs "
        f"{crop2} ({confidence2:.1f}% confidence) "
        f"for {season} season on {soil_type} soil.\n\n"
        f"Write 4 sections:\n\n"
        f"1. Season and Soil Fit\n"
        f"Which crop suits better and why.\n\n"
        f"2. Water and Labour\n"
        f"Water needs and labour comparison.\n\n"
        f"3. Profitability\n"
        f"Which gives better income per acre.\n\n"
        f"4. Final Recommendation\n"
        f"Clear winner with reasons and one tip."
    )

    response = client.chat.completions.create(
        model='openai/gpt-oss-20b',
        messages=[
            {
                'role': 'system',
                'content': 'You are a senior farming advisor. Give direct practical crop comparison advice.'
            },
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=1200,
        temperature=0.3
    )

    content = response.choices[0].message.content
    if '<think>' in content:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content.strip()


def get_soil_advice(client, N, P, K, ph, soil_type):

    prompt = (
        f"Give soil improvement advice for:\n"
        f"N={N}, P={P}, K={K} kg/ha, pH={ph}, Soil={soil_type}\n\n"
        f"Write 4 sections:\n\n"
        f"1. Nutrient Correction\n"
        f"Which nutrients need fixing. Products and kg/acre ranges.\n\n"
        f"2. pH Adjustment\n"
        f"Is pH {ph} ideal. What to add and how much.\n\n"
        f"3. Organic Matter\n"
        f"Best organic amendments with quantities. Cover crop suggestion.\n\n"
        f"4. Long Term Plan\n"
        f"Crop rotation, biofertilizers, one easy seasonal practice."
    )

    response = client.chat.completions.create(
        model='openai/gpt-oss-20b',
        messages=[
            {
                'role': 'system',
                'content': 'You are an expert soil scientist. Give specific soil improvement advice with product names and quantities.'
            },
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=1200,
        temperature=0.3
    )

    content = response.choices[0].message.content
    if '<think>' in content:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content.strip()


if __name__ == "__main__":
    print("ai_advice.py ready - Model: openai/gpt-oss-20b")