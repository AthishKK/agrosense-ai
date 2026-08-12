# ================================================
# AgroSense AI — RAG Chatbot
# Uses FAISS + Sentence Transformers + Groq
# Knowledge Base: Built-in Farming Text
# ================================================

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq


# ─────────────────────────────────────────────────
# 1. FARMING KNOWLEDGE BASE
# Source: ICAR, FAO, TNAU guidelines
# ─────────────────────────────────────────────────

FARMING_KNOWLEDGE = [

    # RICE
    "Rice is a Kharif crop that grows best in waterlogged conditions. "
    "It requires high rainfall of 1000-2000mm per year. "
    "Best soil types are clay and loamy soils that retain water well. "
    "Ideal temperature range is 20-35 degrees Celsius. "
    "Rice needs nitrogen fertilizer in two splits: at transplanting and at tillering stage.",

    "Common rice diseases include Blast disease, Brown Spot, and Bacterial Leaf Blight. "
    "Blast disease occurs in high humidity above 80% and temperature above 25 degrees. "
    "Apply Tricyclazole fungicide to control Blast disease. "
    "Brown Spot is caused by nutrient deficiency especially potassium. "
    "Bacterial Leaf Blight spreads in flood conditions and heavy rainfall.",

    "Rice cultivation steps: prepare land by plowing 2-3 times, "
    "apply basal dose of fertilizer before transplanting, "
    "transplant 25-30 day old seedlings at 20x15 cm spacing, "
    "maintain 2-5 cm water level during vegetative stage, "
    "drain field 10 days before harvest for easy harvesting.",

    "Rice fertilizer recommendation: Apply 120 kg Nitrogen, 60 kg Phosphorus, "
    "60 kg Potassium per hectare for high yielding varieties. "
    "Split nitrogen into 3 doses: 40% basal, 30% at tillering, 30% at panicle initiation. "
    "Use Urea for nitrogen, DAP for phosphorus, MOP for potassium.",

    # WHEAT
    "Wheat is a Rabi crop grown in cool dry weather. "
    "It requires temperature of 10-25 degrees Celsius during growing season. "
    "Ideal rainfall is 25-75 cm well distributed. "
    "Best soils are well drained loamy and clay loamy soils. "
    "Wheat is sown from October to December in India.",

    "Wheat diseases include Yellow Rust, Brown Rust, and Powdery Mildew. "
    "Yellow Rust appears as yellow stripes on leaves in cool humid conditions. "
    "Apply Propiconazole fungicide at first sign of rust infection. "
    "Powdery Mildew appears as white powder on leaves in moderate temperature. "
    "Use resistant varieties to reduce disease risk.",

    "Wheat fertilizer recommendation: Apply 120 kg Nitrogen, 60 kg Phosphorus, "
    "40 kg Potassium per hectare. "
    "Apply full phosphorus and potassium at sowing time. "
    "Split nitrogen: 50% at sowing, 25% at first irrigation, 25% at second irrigation. "
    "Irrigate wheat 4-6 times during growing season.",

    # MAIZE
    "Maize is a Kharif crop that grows in warm humid conditions. "
    "It requires temperature of 21-30 degrees Celsius. "
    "Maize needs well drained fertile loamy soil with good organic matter. "
    "Ideal rainfall is 500-750mm during growing season. "
    "Maize is ready to harvest in 90-120 days after sowing.",

    "Maize fertilizer recommendation: Apply 150 kg Nitrogen, 75 kg Phosphorus, "
    "75 kg Potassium per hectare. "
    "Apply full phosphorus and potassium at sowing. "
    "Split nitrogen into 3 doses for best results. "
    "Maize responds very well to zinc fertilizer application.",

    "Common maize pests include stem borer, fall armyworm, and aphids. "
    "Stem borer causes dead heart symptom in young plants. "
    "Apply Chlorpyrifos to control stem borer infestation. "
    "Fall armyworm is a serious pest that eats leaves and can destroy entire crop. "
    "Monitor crop regularly and apply Spinosad for fall armyworm control.",

    # SOIL HEALTH
    "Soil pH is the measure of acidity or alkalinity of soil. "
    "Most crops grow best in pH range of 6.0 to 7.5. "
    "Acidic soil with pH below 6.0 can be corrected by applying agricultural lime. "
    "Alkaline soil with pH above 8.0 can be corrected by applying gypsum or sulphur. "
    "Test soil pH every 2-3 years and adjust accordingly.",

    "Nitrogen is the most important nutrient for crop growth. "
    "Nitrogen deficiency causes yellowing of leaves starting from older leaves. "
    "Apply Urea or Ammonium Sulphate to correct nitrogen deficiency. "
    "Excess nitrogen causes excessive vegetative growth and lodging. "
    "Legume crops like chickpea and lentil fix atmospheric nitrogen naturally.",

    "Phosphorus is essential for root development and flowering. "
    "Phosphorus deficiency causes purple coloration of leaves and poor root growth. "
    "Apply DAP or Single Super Phosphate to correct phosphorus deficiency. "
    "Phosphorus is most available to plants at pH 6.5 to 7.0. "
    "Apply phosphorus fertilizer close to seed or root zone for best results.",

    "Potassium improves crop quality, disease resistance and water use efficiency. "
    "Potassium deficiency causes scorching of leaf edges and weak stems. "
    "Apply Muriate of Potash MOP to correct potassium deficiency. "
    "Sandy soils are more prone to potassium deficiency than clay soils. "
    "Potassium helps crops tolerate drought and disease stress.",

    "Organic matter improves soil structure, water holding capacity and nutrient supply. "
    "Add farmyard manure at 10-15 tonnes per hectare before planting. "
    "Green manure crops like Dhaincha and Sunhemp improve soil fertility. "
    "Compost made from crop residues is an excellent organic fertilizer. "
    "Regular addition of organic matter builds long term soil health.",

    # FERTILIZERS
    "Urea contains 46% nitrogen and is the most common nitrogen fertilizer in India. "
    "Apply Urea in split doses to reduce nitrogen loss. "
    "Do not apply Urea in waterlogged conditions as it causes nitrogen loss. "
    "Neem coated Urea reduces nitrogen loss and improves efficiency. "
    "Urea is best applied when soil has adequate moisture.",

    "DAP Diammonium Phosphate contains 18% nitrogen and 46% phosphorus. "
    "DAP is the most popular phosphorus fertilizer used in India. "
    "Apply DAP at the time of sowing for best results. "
    "DAP also supplies some nitrogen to the crop. "
    "Store DAP in dry place to prevent moisture absorption.",

    "MOP Muriate of Potash contains 60% potassium and is the main potassium fertilizer. "
    "Apply MOP at sowing time mixed with soil. "
    "MOP is suitable for most crops except tobacco and potato. "
    "For tobacco and potato use Sulphate of Potash instead of MOP. "
    "MOP improves crop quality and shelf life of fruits and vegetables.",

    "20-20 fertilizer contains equal parts nitrogen phosphorus and potassium. "
    "It is a balanced fertilizer suitable for most crops. "
    "Apply 20-20 fertilizer at planting time for good establishment. "
    "It is especially useful when all three nutrients are deficient. "
    "Follow soil test recommendations for correct dose.",

    # IRRIGATION
    "Drip irrigation saves 40-50% water compared to flood irrigation. "
    "It delivers water directly to root zone reducing weed growth. "
    "Drip irrigation is best for fruits vegetables and plantation crops. "
    "Install drip system before planting for best results. "
    "Government provides subsidy for drip irrigation installation in India.",

    "Critical irrigation stages for rice are tillering, panicle initiation and flowering. "
    "Rice field should have 2-5 cm standing water during vegetative stage. "
    "Drain rice field completely 10 days before harvest. "
    "Alternate wetting and drying AWD method saves 30% water in rice. "
    "Never let rice field dry out completely during flowering stage.",

    "Wheat requires 4-6 irrigations during the growing season. "
    "Critical irrigation stages are crown root initiation, tillering, "
    "jointing, flowering and grain filling. "
    "First irrigation should be given 20-25 days after sowing. "
    "Avoid irrigation during windy conditions to prevent lodging.",

    # PEST MANAGEMENT
    "Integrated Pest Management IPM combines biological chemical and cultural methods. "
    "Use resistant crop varieties as first line of defense against pests. "
    "Monitor crops regularly to detect pest problems early. "
    "Use biological controls like Trichoderma and Pseudomonas before chemicals. "
    "Apply chemical pesticides only when pest population crosses economic threshold.",

    "Common soil pests include white grubs termites and nematodes. "
    "White grubs damage roots of many crops especially sugarcane and maize. "
    "Apply Chlorpyrifos to soil before planting to control white grubs. "
    "Termites are serious pests in dry regions damaging roots and stems. "
    "Nematodes cause root knot disease reducing water and nutrient uptake.",

    # CROP ROTATION
    "Crop rotation improves soil health and reduces pest and disease problems. "
    "Rotate cereals with legumes to improve soil nitrogen. "
    "Rice wheat rotation is the most common rotation in North India. "
    "Include a legume crop like chickpea or lentil every 3 years. "
    "Avoid growing the same crop family in the same field consecutively.",

    # SUGARCANE
    "Sugarcane is a long duration crop taking 12-18 months to mature. "
    "It requires hot humid climate with temperature of 20-35 degrees. "
    "Sugarcane needs well drained deep fertile loamy soil. "
    "Apply 250 kg Nitrogen 100 kg Phosphorus 120 kg Potassium per hectare. "
    "Sugarcane ratoon crop saves cost of planting in second year.",

    # COTTON
    "Cotton is a Kharif crop requiring hot dry climate for good fiber quality. "
    "It grows best in black cotton soil also called Vertisol. "
    "Cotton requires temperature of 21-35 degrees and 500-1000mm rainfall. "
    "Apply 120 kg Nitrogen 60 kg Phosphorus 60 kg Potassium per hectare. "
    "Bollworm is the most serious pest of cotton causing heavy yield loss.",

    # GENERAL FARMING
    "Soil testing should be done every 2-3 years before planting season. "
    "Collect soil samples from 0-15 cm depth for testing. "
    "Take samples from multiple spots in the field and mix them. "
    "Send samples to soil testing laboratory for NPK and pH analysis. "
    "Follow soil test based fertilizer recommendations for best results.",

    "Seed treatment before sowing protects seeds from soil borne diseases. "
    "Treat seeds with Thiram or Carbendazim fungicide before sowing. "
    "Treat seeds with Rhizobium culture for legume crops. "
    "Treat seeds with Trichoderma for protection against root rot diseases. "
    "Seed treatment is a low cost high return practice for all crops.",

    "Government schemes for farmers in India include PM Kisan Samman Nidhi, "
    "Pradhan Mantri Fasal Bima Yojana crop insurance scheme, "
    "Soil Health Card scheme for free soil testing, "
    "PM Krishi Sinchai Yojana for irrigation support, "
    "Kisan Credit Card for easy agricultural loans.",

    # HORTICULTURE
    "Tomato grows best in temperature of 20-27 degrees Celsius. "
    "It requires well drained fertile loamy soil with pH 6.0-7.0. "
    "Apply 120 kg Nitrogen 60 kg Phosphorus 60 kg Potassium per hectare. "
    "Common tomato diseases are early blight late blight and leaf curl virus. "
    "Stake tomato plants to prevent lodging and improve air circulation.",

    "Banana requires hot humid climate with temperature of 20-35 degrees. "
    "It grows best in well drained deep fertile loamy soil. "
    "Banana needs high rainfall or regular irrigation of 1200-2200mm per year. "
    "Apply 200 kg Nitrogen 60 kg Phosphorus 300 kg Potassium per hectare. "
    "Panama wilt is the most serious disease of banana with no chemical cure.",

    "Mango grows best in tropical climate with distinct dry season. "
    "It requires temperature of 24-27 degrees for good flowering. "
    "Mango needs deep well drained alluvial or laterite soil. "
    "Apply fertilizer after harvest and before flowering for best results. "
    "Mango malformation disease reduces flowering and yield significantly.",

    # WATERMELON
    "Watermelon grows best in hot dry climate with temperature between 25-35 degrees Celsius. "
    "It requires well drained sandy loam or loamy soil with pH between 6.0 and 7.0. "
    "Watermelon needs low to moderate rainfall of 400-600mm and cannot tolerate waterlogging. "
    "It is a Zaid season crop grown between March and June in India. "
    "Apply 100-120 kg Nitrogen, 50-60 kg Phosphorus, 50-60 kg Potassium per hectare. "
    "Common problems include Fusarium wilt, aphids and fruit fly. "
    "Watermelon is ready to harvest 70-90 days after sowing when tendril near fruit dries up.",

    # MUSKMELON
    "Muskmelon grows best in hot dry climate with temperature between 25-35 degrees Celsius. "
    "It requires well drained sandy loam soil with pH between 6.0 and 7.0. "
    "Muskmelon is a Zaid season crop needing low rainfall of 300-500mm. "
    "Apply 80-100 kg Nitrogen, 40-60 kg Phosphorus, 40-60 kg Potassium per hectare. "
    "Muskmelon needs frequent light irrigation especially during fruit development stage. "
    "Common diseases include powdery mildew and downy mildew. "
    "Fruits are ready when they emit sweet fragrance and slip easily from vine.",

    # APPLE
    "Apple grows best in cool temperate climate with temperature between 10-25 degrees Celsius. "
    "It requires well drained deep loamy soil with pH between 5.5 and 6.5. "
    "Apple needs chilling hours below 7 degrees for proper flowering and fruiting. "
    "Annual rainfall of 1000-1250mm well distributed throughout the year is ideal. "
    "Apply 70-80 kg Nitrogen, 35-40 kg Phosphorus, 35-40 kg Potassium per tree per year. "
    "Common diseases include scab, powdery mildew and fire blight. "
    "Apple is mainly grown in Himachal Pradesh, Jammu Kashmir and Uttarakhand in India.",

    # GRAPES
    "Grapes grow best in hot dry climate with temperature between 15-40 degrees Celsius. "
    "They require well drained deep loamy or sandy loam soil with pH between 6.5 and 7.5. "
    "Grapes need low rainfall of 500-900mm and cannot tolerate waterlogging. "
    "Apply 150-200 kg Nitrogen, 60-80 kg Phosphorus, 100-150 kg Potassium per hectare. "
    "Grapes need training on trellis or bower system. "
    "Common diseases include powdery mildew, downy mildew and anthracnose. "
    "Maharashtra and Karnataka are major grape growing states in India.",

    # ORANGE
    "Orange grows best in subtropical climate with temperature between 13-38 degrees Celsius. "
    "It requires well drained deep loamy soil with pH between 5.5 and 7.5. "
    "Orange needs moderate rainfall of 750-1250mm per year. "
    "Apply 600-900 grams Nitrogen, 200-300 grams Phosphorus, 200-400 grams Potassium per tree per year. "
    "Common diseases include citrus canker, greening disease and gummosis. "
    "Nagpur in Maharashtra is famous for orange production in India.",

    # PAPAYA
    "Papaya grows best in tropical climate with temperature between 22-26 degrees Celsius. "
    "It requires well drained fertile loamy soil with pH between 6.0 and 7.0. "
    "Papaya needs moderate rainfall of 1000-1500mm and cannot tolerate waterlogging. "
    "Apply 200-250 kg Nitrogen, 200-250 kg Phosphorus, 200-250 kg Potassium per hectare per year. "
    "Papaya starts bearing fruit within 9-12 months of planting. "
    "Common diseases include papaya ring spot virus, powdery mildew and anthracnose.",

    # POMEGRANATE
    "Pomegranate grows best in hot dry climate with temperature between 25-35 degrees Celsius. "
    "It requires well drained loamy or sandy loam soil with pH between 5.5 and 7.2. "
    "Pomegranate is drought tolerant and needs low rainfall of 500-800mm. "
    "Apply 125-250 grams Nitrogen per plant per year split into multiple doses. "
    "Pomegranate is highly profitable and suitable for dry regions of Maharashtra, Karnataka, Gujarat and Rajasthan. "
    "Common diseases include bacterial blight and fruit rot.",

    # COCONUT
    "Coconut grows best in tropical coastal climate with temperature between 27-32 degrees Celsius. "
    "It requires well drained sandy loam or laterite soil with pH between 5.2 and 8.0. "
    "Coconut needs high rainfall of 1000-3000mm well distributed. "
    "Apply 500 grams Nitrogen, 320 grams Phosphorus, 1200 grams Potassium per palm per year. "
    "Kerala, Tamil Nadu and Karnataka are major coconut growing states. "
    "Common pests include rhinoceros beetle, red palm weevil and eriophyid mite.",

    # JUTE
    "Jute grows best in hot humid climate with temperature between 24-38 degrees Celsius. "
    "It requires well drained loamy or sandy loam soil with pH between 6.0 and 7.5. "
    "Jute needs high rainfall of 1000-2000mm during growing season. "
    "It is a Kharif crop sown from March to May in India. "
    "Apply 60-80 kg Nitrogen, 30-40 kg Phosphorus, 30-40 kg Potassium per hectare. "
    "West Bengal and Bihar are major jute producing states. "
    "Common diseases include stem rot, leaf spot and powdery mildew.",

    # CHICKPEA
    "Chickpea is a Rabi season legume crop grown in cool dry weather. "
    "It requires well drained loamy or sandy loam soil with pH between 6.0 and 9.0. "
    "Chickpea needs low rainfall of 300-500mm and is highly drought tolerant. "
    "Being a legume it fixes atmospheric nitrogen improving soil fertility naturally. "
    "Apply 20-30 kg Nitrogen, 40-60 kg Phosphorus, 20-30 kg Potassium per hectare. "
    "Common diseases include wilt, blight and pod borer which is the most serious pest.",

    # LENTIL
    "Lentil is a Rabi season pulse crop grown in cool dry climate. "
    "It requires well drained loamy soil with pH between 6.0 and 8.0. "
    "Lentil needs very low rainfall of 250-400mm and is drought tolerant. "
    "Being a legume it fixes atmospheric nitrogen and improves soil health. "
    "Apply 20 kg Nitrogen, 40-60 kg Phosphorus, 20-30 kg Potassium per hectare. "
    "Common diseases include rust, wilt and stemphylium blight.",

    # KIDNEY BEANS
    "Kidney beans grow best in cool to warm climate with temperature between 18-24 degrees Celsius. "
    "They require well drained fertile loamy soil with pH between 6.0 and 7.5. "
    "Kidney beans need moderate rainfall of 300-400mm during growing season. "
    "Being a legume they fix atmospheric nitrogen naturally improving soil fertility. "
    "Apply 20-25 kg Nitrogen, 50-60 kg Phosphorus, 30-40 kg Potassium per hectare. "
    "Common diseases include angular leaf spot, bean rust and anthracnose.",

    # BLACKGRAM
    "Blackgram is a Kharif pulse crop growing best in hot humid conditions. "
    "It requires well drained loamy or clay loam soil with pH between 6.0 and 7.5. "
    "Blackgram needs moderate rainfall of 600-1000mm during growing season. "
    "Being a legume it fixes atmospheric nitrogen enriching soil naturally. "
    "Apply 20-25 kg Nitrogen, 40-50 kg Phosphorus, 20-25 kg Potassium per hectare. "
    "Common diseases include yellow mosaic virus, powdery mildew and leaf crinkle.",

    # MUNGBEAN
    "Mungbean is a warm season pulse crop grown in Kharif and Zaid seasons. "
    "It requires well drained sandy loam soil with pH between 6.2 and 7.2. "
    "Mungbean needs moderate rainfall of 600-900mm and matures in 60-90 days. "
    "Being a legume it fixes atmospheric nitrogen improving soil fertility. "
    "Apply 20 kg Nitrogen, 40-50 kg Phosphorus, 20-30 kg Potassium per hectare. "
    "Common diseases include yellow mosaic virus and cercospora leaf spot.",

    # MOTHBEANS
    "Mothbeans are highly drought tolerant warm season pulse crop. "
    "They grow best in arid and semi-arid regions with temperature between 24-32 degrees Celsius. "
    "Mothbeans require well drained sandy or sandy loam soil with pH between 6.0 and 8.0. "
    "They need very low rainfall of 300-400mm and are suitable for dry regions of Rajasthan. "
    "Being a legume they fix atmospheric nitrogen naturally. "
    "Apply 20 kg Nitrogen, 40 kg Phosphorus per hectare at sowing time.",

    # PIGEONPEAS
    "Pigeonpeas also called Tur or Arhar is a Kharif pulse crop. "
    "It grows best in warm climate with temperature between 18-38 degrees Celsius. "
    "Pigeonpeas require well drained loamy or clay loam soil with pH between 5.0 and 7.5. "
    "They need moderate rainfall of 600-1000mm and are moderately drought tolerant. "
    "Being a deep rooted legume they improve soil structure and fix nitrogen. "
    "Apply 20-25 kg Nitrogen, 50-60 kg Phosphorus, 20-30 kg Potassium per hectare. "
    "Common pest is pod borer the most serious threat.",

    # BANANA (detailed)
    "Banana requires hot humid climate with temperature between 20-35 degrees Celsius. "
    "It grows best in well drained deep fertile loamy soil with pH between 6.0 and 7.5. "
    "Banana needs high rainfall or regular irrigation of 1200-2200mm per year. "
    "Apply 200 kg Nitrogen, 60 kg Phosphorus, 300 kg Potassium per hectare. "
    "Panama wilt is the most serious disease of banana with no chemical cure. "
    "Common pests include banana weevil and nematodes. "
    "Banana is ready to harvest 12-15 months after planting.",

    # COFFEE
    "Coffee grows best in tropical highland climate with temperature between 15-28 degrees Celsius. "
    "It requires well drained deep fertile loamy soil with pH between 6.0 and 6.5. "
    "Coffee needs high rainfall of 1500-2500mm well distributed. "
    "Apply 40-50 kg Nitrogen, 20-30 kg Phosphorus, 40-50 kg Potassium per hectare per year. "
    "Coffee leaf rust is the most serious disease controlled by copper based fungicides. "
    "Karnataka produces 70 percent of India total coffee production. "
    "Coffee takes 3-4 years after planting to start bearing fruit.",
]


# ─────────────────────────────────────────────────
# 2. BUILD FAISS KNOWLEDGE BASE
# ─────────────────────────────────────────────────

def build_knowledge_base(save_path='models/faiss_index.pkl'):
    """
    Converts farming knowledge text into vectors
    and saves FAISS index
    """
    print("Loading sentence transformer model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    print("Creating embeddings for knowledge base...")
    embeddings = embedder.encode(FARMING_KNOWLEDGE, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save everything
    knowledge_store = {
        'index': index,
        'chunks': FARMING_KNOWLEDGE,
        'embeddings': embeddings
    }

    with open(save_path, 'wb') as f:
        pickle.dump(knowledge_store, f)

    print(f"✅ Knowledge base built!")
    print(f"   Total chunks: {len(FARMING_KNOWLEDGE)}")
    print(f"   Embedding dimension: {dimension}")
    print(f"   Saved to: {save_path}")

    return knowledge_store


# ─────────────────────────────────────────────────
# 3. LOAD KNOWLEDGE BASE
# ─────────────────────────────────────────────────

def load_knowledge_base(save_path='models/faiss_index.pkl'):
    """Load existing FAISS knowledge base"""
    if not os.path.exists(save_path):
        print("Knowledge base not found. Building now...")
        return build_knowledge_base(save_path)

    with open(save_path, 'rb') as f:
        knowledge_store = pickle.load(f)

    print(f"✅ Knowledge base loaded!")
    print(f"   Total chunks: {len(knowledge_store['chunks'])}")
    return knowledge_store


# ─────────────────────────────────────────────────
# 4. RETRIEVE RELEVANT CHUNKS
# ─────────────────────────────────────────────────

def retrieve_chunks(query, knowledge_store, embedder, top_k=3):
    """
    Find most relevant knowledge chunks for a query
    """
    query_embedding = embedder.encode([query]).astype('float32')
    distances, indices = knowledge_store['index'].search(
        query_embedding, top_k
    )

    retrieved = []
    for idx in indices[0]:
        if idx < len(knowledge_store['chunks']):
            retrieved.append(knowledge_store['chunks'][idx])

    return retrieved


# ─────────────────────────────────────────────────
# 5. ANSWER QUESTION USING RAG
# ─────────────────────────────────────────────────

def answer_question(question, knowledge_store,
                    embedder, groq_client,
                    farm_context=""):
    """
    Full RAG pipeline with structured output and optional farm context.
    Question → Retrieve chunks → Send to Groq → Structured Answer
    """

    # Step 1: Retrieve relevant chunks
    relevant_chunks = retrieve_chunks(
        question, knowledge_store, embedder, top_k=3
    )

    # Step 2: Build context
    context = "\n\n".join(relevant_chunks)

    # Step 3: Build prompt with structured output instruction
    farm_ctx_block = f"\n\nFARMER'S FARM DATA:\n{farm_context}" if farm_context else ""

    prompt = f"""You are AgroSense AI, an expert agricultural advisor for Indian farmers.
Use the farming knowledge below to answer the question.
If the answer is not in the knowledge, say: "I don't have specific information about that — please consult your local agricultural extension officer."
{farm_ctx_block}

FARMING KNOWLEDGE:
{context}

FARMER'S QUESTION: {question}

Respond in this EXACT structured format:

[ANSWER]
Write 2-3 clear sentences directly answering the question. If farm data is available above, personalise the answer (e.g. "Based on your Rice recommendation...").

[STEPS]
• Step or tip 1
• Step or tip 2
• Step or tip 3
• Step or tip 4 (if needed)

[PRO TIP]
One expert pro tip the farmer should know.

Use simple language. Be specific with product names and quantities where relevant.
"""

    # Step 4: Get answer from Groq
    response = groq_client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {
                'role': 'system',
                'content': (
                    'You are AgroSense AI, a friendly and expert agricultural advisor. '
                    'Always respond using the exact [ANSWER], [STEPS], [PRO TIP] format. '
                    'Be practical, specific, and use simple language farmers can understand.'
                )
            },
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=900,
        temperature=0.65
    )

    raw = response.choices[0].message.content

    # Step 5: Parse structured sections into formatted HTML-friendly text
    import re
    answer_match = re.search(r'\[ANSWER\]\s*(.+?)(?=\[STEPS\]|\[PRO TIP\]|$)', raw, re.DOTALL)
    steps_match  = re.search(r'\[STEPS\]\s*(.+?)(?=\[PRO TIP\]|$)', raw, re.DOTALL)
    tip_match    = re.search(r'\[PRO TIP\]\s*(.+?)$', raw, re.DOTALL)

    answer_text = answer_match.group(1).strip() if answer_match else raw
    steps_text  = steps_match.group(1).strip()  if steps_match  else ""
    tip_text    = tip_match.group(1).strip()     if tip_match    else ""

    # Build final formatted answer
    parts = [answer_text]
    if steps_text:
        parts.append("\n" + steps_text)
    if tip_text:
        parts.append(f"\n\n💡 Pro Tip: {tip_text}")

    formatted_answer = "\n".join(parts)

    return {
        'question': question,
        'answer':   formatted_answer,
        'sources':  relevant_chunks
    }


# ─────────────────────────────────────────────────
# 6. INITIALIZE RAG SYSTEM
# ─────────────────────────────────────────────────

def initialize_rag(groq_api_key,
                   index_path='models/faiss_index.pkl'):
    """
    Initialize complete RAG system
    Returns: knowledge_store, embedder, groq_client
    """
    print("Initializing RAG system...")

    # Load or build knowledge base
    knowledge_store = load_knowledge_base(index_path)

    # Load embedder
    print("Loading embedder...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize Groq
    groq_client = Groq(api_key=groq_api_key)

    print("✅ RAG system ready!")
    return knowledge_store, embedder, groq_client


# ─────────────────────────────────────────────────
# 7. REBUILD INDEX
# ─────────────────────────────────────────────────

def rebuild_index():
    import os
    index_path = 'models/faiss_index.pkl'
    if os.path.exists(index_path):
        os.remove(index_path)
        print('Old index deleted')
    build_knowledge_base(index_path)
    print('Knowledge base rebuilt with all crops!')

if __name__ == '__main__':
    rebuild_index()