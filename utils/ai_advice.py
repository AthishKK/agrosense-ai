import sys
sys.stdout.reconfigure(encoding='utf-8')
# ================================================
# AgroSense AI — AI Farming Advice
# Uses Groq API (Free, No Daily Limit)
# Model: llama-3.3-70b-versatile
# ================================================

from groq import Groq


# ─────────────────────────────────────────────────
# CONFIGURE GROQ
# ─────────────────────────────────────────────────

def configure_groq(api_key):
    """Initialize Groq client with API key"""
    client = Groq(api_key=api_key)
    return client


# ─────────────────────────────────────────────────
# 1. FARMING ADVICE AFTER PREDICTION
# ─────────────────────────────────────────────────

def get_farming_advice(client, crop, fertilizer, yield_range,
                       N, P, K, ph, temperature, humidity,
                       rainfall, soil_type, season):
    """
    Generates personalized farming advice based on
    prediction results and input conditions
    """

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

    prompt = f"""A farmer's AI analysis has produced the following data:

━━━ FARM DATA ━━━
Crop Recommended : {crop}
Fertilizer       : {fertilizer}
Expected Yield   : {yield_range} tonnes/hectare
Season           : {season}
Soil Type        : {soil_type}
Nitrogen (N)     : {N} kg/ha
Phosphorus (P)   : {P} kg/ha
Potassium (K)    : {K} kg/ha
Soil pH          : {ph}
Temperature      : {temperature}°C
Humidity         : {humidity}%
Rainfall         : {rainfall} mm/year
Disease Risk     : {disease_risk_note}
━━━━━━━━━━━━━━━━━

Write a complete farming guide with EXACTLY these 6 section headings:

## SEED SELECTION
## FERTILIZER SCHEDULE
## IRRIGATION PLAN
## DISEASE & PEST PREVENTION
## EXPECTED CHALLENGES
## HARVEST GUIDANCE

STRICT RULES — follow every rule without exception:
- NEVER give specific rupee amounts anywhere in the response.
- NEVER say a disease "will appear" — only say "risk is elevated if conditions persist".
- NEVER mention labour costs or hired workers.
- Give ALL fertilizer quantities as RANGES only (e.g. "80-120 kg/ha", never "100 kg/ha").
- Base disease risk ONLY on the Disease Risk note above — do not invent new risks.
- NEVER mention Krishi Vigyan Kendra.
- NEVER mention agmarknet or any external websites.
- Do NOT include a Soil Preparation section.
- Do NOT include a Market Guidance section.
- End the entire response with exactly this line:
  "ℹ️ These recommendations are based on verified agronomic guidelines from ICAR and FAO integrated into AgroSense AI. Field conditions vary — monitor your crop regularly and adjust as needed."

Section-specific guidance:

## SEED SELECTION
- 2-3 best {crop} varieties for {season} and {soil_type} soil
- Seed rate as range (kg/acre), seed treatment method
- Sowing depth and spacing (row x plant in cm)
- Germination tips for {temperature}°C

## FERTILIZER SCHEDULE
- Week-by-week timeline from sowing to harvest
- All doses as ranges (kg/acre)
- Micronutrient corrections for pH={ph}
- Foliar spray options with dilution ratios

## IRRIGATION PLAN
- Days after sowing for first irrigation
- Frequency and quantity per growth stage
- Critical stress stages for {crop}
- Water-saving tips for {rainfall} mm/year rainfall

## DISEASE & PEST PREVENTION
- Use ONLY the disease risk note provided above
- Top 2-3 pests relevant to {crop} at {temperature}°C
- Preventive spray options with dose ranges
- Organic alternatives

## EXPECTED CHALLENGES
- Top 3 challenges based on temp={temperature}°C, humidity={humidity}%,
  rainfall={rainfall}mm, pH={ph}
- Detection and mitigation for each

## HARVEST GUIDANCE
- Days-after-sowing range for {crop} in {season}
- 4-5 visual maturity signs
- Post-harvest handling: drying, storage method and duration
"""

    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {
                'role': 'system',
                'content': (
                    'You are a careful agricultural advisor with expertise in Indian farming. '
                    'You never invent specific prices, exact quantities or guaranteed predictions. '
                    'You give practical ranges and evidence-based guidance only.'
                )
            },
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=2500,
        temperature=0.3
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────
# 2. CROP COMPARISON ADVICE
# ─────────────────────────────────────────────────

def get_comparison_advice(client, crop1, crop2, confidence1,
                          confidence2, season, soil_type):
    """
    Compares two crops and gives recommendation
    """

    prompt = f"""
You are a senior agricultural advisor in India with 25 years of experience.
A farmer is deciding between two crops recommended by an AI system:

Crop 1: {crop1} (AI confidence: {confidence1:.1f}%)
Crop 2: {crop2} (AI confidence: {confidence2:.1f}%)
Season: {season}
Soil Type: {soil_type}

Give a detailed comparison covering these 4 points:

1. SEASON & SOIL FIT
- Which crop is better suited for {season} season in {soil_type} soil and why
- Risk of choosing the lower-confidence crop

2. WATER & LABOUR REQUIREMENTS
- Water needs comparison (litres/acre/week) for both crops
- Labour intensity and cost difference in ₹/acre

3. MARKET DEMAND & PROFITABILITY
- Current market demand and average price (₹/quintal) for both crops
- Which gives better net profit per acre and why
- Storage and perishability comparison

4. FINAL RECOMMENDATION
- Clear winner with specific reasons
- One risk to watch out for with the recommended crop
- One tip to maximise profit from the chosen crop

Be specific, practical and use simple language.
Format as numbered points 1-4 with bullet points under each.
"""

    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {
                'role': 'system',
                'content': 'You are a senior agricultural advisor. Give detailed, specific crop comparison advice with quantities and prices. Never skip any section.'
            },
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=1500,
        temperature=0.7
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────
# 3. SOIL IMPROVEMENT ADVICE
# ─────────────────────────────────────────────────

def get_soil_advice(client, N, P, K, ph, soil_type):
    """
    Gives advice on how to improve soil health
    """

    prompt = f"""
You are an expert soil scientist and agricultural advisor in India.
A farmer's soil test shows:

- Nitrogen (N): {N} kg/ha
- Phosphorus (P): {P} kg/ha
- Potassium (K): {K} kg/ha
- pH: {ph}
- Soil Type: {soil_type}

Give a detailed soil improvement plan covering these 4 sections:

1. NUTRIENT CORRECTION
- Identify which nutrients are deficient, sufficient or excess
- Exact fertilizer products and kg/acre doses to correct each deficiency
- Application method (broadcast, band, foliar) and timing

2. pH ADJUSTMENT
- Is the pH of {ph} ideal, acidic or alkaline for most crops?
- If correction needed: product name (lime/gypsum/sulphur), exact kg/acre dose
- How long before sowing to apply the pH amendment

3. ORGANIC MATTER & SOIL STRUCTURE
- Recommended organic amendments for {soil_type} soil (FYM, vermicompost,
  green manure) with kg/acre quantities
- How to improve water retention or drainage based on soil type
- Cover crop or green manure suggestion between seasons

4. LONG-TERM SOIL HEALTH PLAN
- Crop rotation sequence to maintain soil fertility
- Biofertilizer recommendations (Rhizobium, PSB, Azotobacter) with doses
- One low-cost practice the farmer can do every season to keep soil healthy
- Expected improvement in yield after following this plan for 2 seasons

Be specific, practical and affordable for small farmers.
Format as numbered points 1-4 with bullet points under each.
"""

    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {
                'role': 'system',
                'content': 'You are an expert soil scientist. Give detailed, specific soil improvement advice with exact product names, quantities and timings. Never skip any section.'
            },
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=1500,
        temperature=0.7
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────
# 4. QUICK TEST
# ─────────────────────────────────────────────────

if __name__ == "__main__":

    # ⚠️ Replace with your actual Groq API key for testing
    API_KEY = "<your_groq_api_key>"

    print("=" * 50)
    print("Testing Groq AI Farming Advice")
    print("=" * 50)

    # Initialize client
    client = configure_groq(API_KEY)
    print("✅ Groq client initialized!\n")

    # Test 1 — Farming advice
    print("📋 Test 1: Farming Advice")
    print("-" * 40)
    advice = get_farming_advice(
        client=client,
        crop="Rice",
        fertilizer="Urea",
        yield_range="3.8 — 4.5",
        N=90,
        P=42,
        K=43,
        ph=6.5,
        temperature=28,
        humidity=82,
        rainfall=200,
        soil_type="Loamy",
        season="Kharif"
    )
    print(advice)

    # Test 2 — Crop comparison
    print("\n📋 Test 2: Crop Comparison")
    print("-" * 40)
    comparison = get_comparison_advice(
        client=client,
        crop1="Rice",
        crop2="Maize",
        confidence1=90.0,
        confidence2=25.0,
        season="Kharif",
        soil_type="Loamy"
    )
    print(comparison)

    # Test 3 — Soil advice
    print("\n📋 Test 3: Soil Improvement Advice")
    print("-" * 40)
    soil_advice = get_soil_advice(
        client=client,
        N=90,
        P=8,
        K=43,
        ph=6.5,
        soil_type="Loamy"
    )
    print(soil_advice)

    print("\n" + "=" * 50)
    print("✅ All Groq tests passed!")
    print("✅ utils/08_gemini_advice.py is ready!")
    print("=" * 50)