import streamlit as st
import joblib
import numpy as np
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CardioAI — Heart Risk Predictor",
    page_icon="🫀",
    layout="centered"
)

# =========================
# BACKGROUND & STYLES
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');

/* Animated gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 30%, #1a0a2e 60%, #0a1628 100%);
    background-size: 400% 400%;
    animation: bgShift 18s ease infinite;
    font-family: 'Inter', sans-serif;
    color: #e0e0f0;
}

@keyframes bgShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glowing card container */
section[data-testid="stSidebar"] { display: none; }

div[data-testid="stVerticalBlock"] > div {
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 4px;
}

/* Title */
h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.8rem !important;
    background: linear-gradient(90deg, #ff6b9d, #c44dff, #6bc5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0 !important;
}

h2, h3 {
    color: #c0c8ff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}

/* Subheader divider */
.section-label {
    background: linear-gradient(90deg, #ff6b9d33, transparent);
    border-left: 3px solid #ff6b9d;
    padding: 6px 14px;
    border-radius: 0 8px 8px 0;
    color: #ff6b9d;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 16px 0 8px 0;
}

/* Input labels */
label {
    color: #a0aec0 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

/* Input boxes */
input, select, textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Predict button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #ff6b9d, #c44dff, #6bc5ff) !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px !important;
    letter-spacing: 1px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(196, 77, 255, 0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(196, 77, 255, 0.6) !important;
}

/* Result boxes */
.result-high {
    background: linear-gradient(135deg, #2d0a0a, #4a0f0f);
    border: 1px solid #ff4444;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 30px rgba(255,68,68,0.3);
}

.result-moderate {
    background: linear-gradient(135deg, #2d1f0a, #4a350f);
    border: 1px solid #ffaa44;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 30px rgba(255,170,68,0.3);
}

.result-low {
    background: linear-gradient(135deg, #0a2d0a, #0f4a1a);
    border: 1px solid #44ff88;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 30px rgba(68,255,136,0.3);
}

.result-title {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}

.risk-percent {
    font-size: 3.5rem;
    font-weight: 800;
    margin: 8px 0;
}

/* Diet card */
.diet-card {
    background: rgba(107, 197, 255, 0.08);
    border: 1px solid rgba(107, 197, 255, 0.25);
    border-radius: 14px;
    padding: 16px 20px;
    margin: 6px 0;
}

.diet-icon {
    font-size: 1.4rem;
    margin-right: 8px;
}

/* Download button */
.stDownloadButton > button {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 10px !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.1) !important;
}

/* Number inputs */
div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.05) !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #44ff88, #ffaa44, #ff4444) !important;
}

/* Footer */
.footer {
    text-align: center;
    color: rgba(255,255,255,0.25);
    font-size: 0.75rem;
    margin-top: 30px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<h1>🫀 CardioAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#7a8aa0; margin-top:4px; font-size:0.95rem;'>Heart Disease Risk Prediction System</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.warning("⚠️ Model file not found. Using demo mode.")

# =========================
# PATIENT INFO
# =========================
st.markdown('<div class="section-label">👤 Patient Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Full Name", placeholder="e.g. Ravi Kumar")
    age = st.number_input("Age", 1, 120, 45)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    sex = 1 if gender == "Male" else 0
    doctor = st.text_input("Doctor / Referred By", placeholder="e.g. Dr. Sharma")

# =========================
# CLINICAL DATA
# =========================
st.markdown('<div class="section-label">🩺 Clinical Measurements</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    trestbps = st.number_input("Blood Pressure (mmHg)", 80, 250, 120)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    thalach = st.number_input("Max Heart Rate (bpm)", 60, 250, 150)
    oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)

with col4:
    fbs_input = st.selectbox("Fasting Blood Sugar", ["Normal (< 120 mg/dL)", "High (≥ 120 mg/dL)"])
    fbs = 1 if "High" in fbs_input else 0

    restecg_input = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    restecg = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg_input]

    exang_input = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    exang = 1 if exang_input == "Yes" else 0

    slope_input = st.selectbox("ST Slope", ["Upsloping (Good)", "Flat (Medium)", "Downsloping (High Risk)"])
    slope = {"Upsloping (Good)": 0, "Flat (Medium)": 1, "Downsloping (High Risk)": 2}[slope_input]

# =========================
# ADVANCED
# =========================
st.markdown('<div class="section-label">🔬 Advanced Parameters</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    cp_map = {
        "No Chest Pain": 3,
        "Typical Angina (during activity)": 0,
        "Atypical Angina": 1,
        "Non-Anginal Pain": 2
    }
    cp = cp_map[st.selectbox("Chest Pain Type", list(cp_map.keys()))]
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0–4)", [0, 1, 2, 3, 4])

with col6:
    thal_input = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"])
    thal = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3, "Unknown": 0}[thal_input]

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# DIET PLANS
# =========================
def get_diet_plan(prob_pct):
    if prob_pct >= 70:
        return {
            "color": "#ff4444",
            "emoji": "🔴",
            "label": "HIGH RISK",
            "foods_eat": [
                ("🥦", "Leafy greens & cruciferous vegetables (spinach, broccoli, kale)"),
                ("🐟", "Fatty fish — omega-3 rich (salmon, mackerel, sardines)"),
                ("🫐", "Berries & antioxidant-rich fruits (blueberries, pomegranate)"),
                ("🥣", "Oats, barley — beta-glucan to lower LDL cholesterol"),
                ("🧄", "Garlic & turmeric — natural anti-inflammatory agents"),
                ("🥜", "Walnuts, flaxseeds — heart-protective fats"),
            ],
            "foods_avoid": [
                ("🍟", "Fried & deep-fried foods — raise bad LDL cholesterol"),
                ("🧂", "High-sodium foods, pickles, processed snacks"),
                ("🥩", "Red meat & organ meats — saturated fat overload"),
                ("🥤", "Sugary drinks, soda, packaged juices"),
                ("🍰", "Trans fats — margarine, pastries, fast food"),
                ("🚬", "Smoking & alcohol — direct cardiac damage"),
            ],
            "lifestyle": [
                "🏥 Consult a cardiologist immediately",
                "💊 Strictly follow prescribed medications",
                "🚶 Light walks only — 15–20 min, doctor-supervised",
                "🧘 Stress reduction: yoga, deep breathing daily",
                "😴 Sleep 7–8 hours — poor sleep worsens heart health",
                "📊 Monitor BP & heart rate daily",
            ],
            "meals": {
                "Breakfast": "Oatmeal with walnuts + green tea (no sugar)",
                "Mid-Morning": "Handful of almonds + 1 apple",
                "Lunch": "Brown rice + palak dal + cucumber salad + buttermilk",
                "Evening": "Roasted chana + herbal tea",
                "Dinner": "Grilled fish / tofu + steamed vegetables + small roti",
            }
        }
    elif prob_pct >= 40:
        return {
            "color": "#ffaa44",
            "emoji": "🟡",
            "label": "MODERATE RISK",
            "foods_eat": [
                ("🥗", "Colorful salads with olive oil dressing"),
                ("🍎", "Fresh seasonal fruits — 2–3 servings daily"),
                ("🌾", "Whole grains — brown rice, multigrain roti"),
                ("🥚", "Egg whites — lean protein without saturated fat"),
                ("🫘", "Legumes — rajma, chickpeas, lentils"),
                ("🥛", "Low-fat dairy — curd, buttermilk"),
            ],
            "foods_avoid": [
                ("🍔", "Fast food & junk food — hidden trans fats"),
                ("🧁", "Sweets & desserts — spike blood sugar"),
                ("🥓", "Processed meats — sausages, bacon"),
                ("🧈", "Excess butter & ghee"),
                ("🍺", "Alcohol — raises triglycerides"),
                ("☕", "Excess caffeine — more than 2 cups/day"),
            ],
            "lifestyle": [
                "🏃 30 min moderate exercise — walking, cycling, swimming",
                "⚖️ Maintain healthy BMI (18.5–24.9)",
                "🩺 Annual cardiac checkup",
                "🧘 Meditation — 10 min daily for stress",
                "😴 Regular sleep schedule — same time daily",
                "🚰 Drink 2.5–3 liters of water daily",
            ],
            "meals": {
                "Breakfast": "Vegetable upma / poha + 1 banana + low-fat milk",
                "Mid-Morning": "Fruit bowl or sprouts chaat",
                "Lunch": "2 multigrain rotis + sabzi + dal + salad",
                "Evening": "Green tea + handful of mixed nuts",
                "Dinner": "Vegetable soup + 1 roti + curd",
            }
        }
    else:
        return {
            "color": "#44ff88",
            "emoji": "🟢",
            "label": "LOW RISK",
            "foods_eat": [
                ("🍽️", "Balanced plate — 50% vegetables, 25% protein, 25% grains"),
                ("🥝", "Vitamin C rich fruits — kiwi, orange, guava"),
                ("🌿", "Fresh herbs — coriander, mint in daily cooking"),
                ("🥜", "Mixed nuts — 1 small handful daily"),
                ("🐓", "Lean proteins — chicken, fish, tofu, eggs"),
                ("💧", "Stay hydrated — 8–10 glasses water daily"),
            ],
            "foods_avoid": [
                ("🍿", "Excess processed snacks — even 'healthy' ones"),
                ("🧂", "Hidden sodium in sauces, condiments"),
                ("🥩", "Red meat more than twice a week"),
                ("🍭", "Refined sugar in excess"),
            ],
            "lifestyle": [
                "🏋️ 45 min exercise, 5 days a week",
                "🧘 Maintain stress levels — hobbies, social connections",
                "🩺 Routine checkup every 2 years",
                "🥗 Continue healthy eating habits",
                "😴 Quality sleep — 7–9 hours consistently",
                "🚭 Stay smoke-free and limit alcohol",
            ],
            "meals": {
                "Breakfast": "Idli / dosa with sambar + fresh fruit juice",
                "Mid-Morning": "Greek yogurt or fruit",
                "Lunch": "Rice / roti + dal + sabzi + salad + curd",
                "Evening": "Sprouts / nuts + green tea",
                "Dinner": "Light meal — khichdi / soup + salad",
            }
        }

# =========================
# PDF GENERATOR — PROFESSIONAL
# =========================
def generate_pdf(name, age, gender, doctor, prob, result, diet, input_data):
    file_name = "CardioAI_Heart_Report.pdf"
    doc = SimpleDocTemplate(
        file_name,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=15*mm,
        bottomMargin=15*mm
    )

    PRIMARY = colors.HexColor("#c44dff")
    DARK = colors.HexColor("#0d1b2a")
    ACCENT = colors.HexColor("#ff6b9d")
    SUCCESS = colors.HexColor("#44ff88")
    WARNING = colors.HexColor("#ffaa44")
    DANGER = colors.HexColor("#ff4444")
    LIGHT_BG = colors.HexColor("#f0f4ff")
    TEXT = colors.HexColor("#1a1a2e")
    GRAY = colors.HexColor("#6b7280")

    risk_color = DANGER if prob >= 70 else (WARNING if prob >= 40 else SUCCESS)

    styles = getSampleStyleSheet()

    def sty(name, **kwargs):
        return ParagraphStyle(name, **kwargs)

    title_sty = sty("Title2", fontName="Helvetica-Bold", fontSize=22, textColor=colors.white, alignment=TA_CENTER, spaceAfter=2)
    sub_sty = sty("Sub", fontName="Helvetica", fontSize=10, textColor=colors.HexColor("#c0c8ff"), alignment=TA_CENTER, spaceAfter=4)
    section_sty = sty("Section", fontName="Helvetica-Bold", fontSize=11, textColor=PRIMARY, spaceBefore=10, spaceAfter=4)
    body_sty = sty("Body2", fontName="Helvetica", fontSize=9.5, textColor=TEXT, leading=15, spaceAfter=3)
    small_sty = sty("Small", fontName="Helvetica", fontSize=8.5, textColor=GRAY, leading=13)
    bold_sty = sty("Bold2", fontName="Helvetica-Bold", fontSize=9.5, textColor=TEXT, leading=14)
    risk_sty = sty("Risk", fontName="Helvetica-Bold", fontSize=28, textColor=risk_color, alignment=TA_CENTER)
    percent_sty = sty("Percent", fontName="Helvetica-Bold", fontSize=42, textColor=risk_color, alignment=TA_CENTER)
    white_sty = sty("White", fontName="Helvetica", fontSize=9, textColor=colors.white, leading=14)
    white_bold_sty = sty("WhiteBold", fontName="Helvetica-Bold", fontSize=9, textColor=colors.white, leading=14)

    content = []

    # ---- HEADER BANNER ----
    header_data = [[
        Paragraph("🫀 CardioAI", title_sty),
    ]]
    header_table = Table(header_data, colWidths=[170*mm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK),
        ("ROWPADDING", (0, 0), (-1, -1), 14),
        ("ROUNDEDCORNERS", [8]),
    ]))
    content.append(header_table)
    content.append(Spacer(1, 3*mm))

    sub_data = [[Paragraph("Comprehensive Heart Disease Risk Assessment Report", sub_sty)]]
    sub_table = Table(sub_data, colWidths=[170*mm])
    sub_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1a0a2e")),
        ("ROWPADDING", (0, 0), (-1, -1), 6),
    ]))
    content.append(sub_table)
    content.append(Spacer(1, 5*mm))

    # ---- REPORT META ----
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")
    meta_data = [
        [Paragraph(f"<b>Patient:</b> {name or 'N/A'}", body_sty),
         Paragraph(f"<b>Age / Gender:</b> {age} yrs / {gender}", body_sty),
         Paragraph(f"<b>Date:</b> {now}", small_sty)],
        [Paragraph(f"<b>Referred By:</b> {doctor or 'Self'}", body_sty),
         Paragraph(f"<b>Report ID:</b> CA-{datetime.now().strftime('%Y%m%d%H%M')}", small_sty),
         Paragraph("<b>System:</b> CardioAI v2.0", small_sty)],
    ]
    meta_table = Table(meta_data, colWidths=[60*mm, 60*mm, 50*mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("ROWPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d0d8f0")),
        ("ROUNDEDCORNERS", [6]),
    ]))
    content.append(meta_table)
    content.append(Spacer(1, 5*mm))

    # ---- RISK RESULT ----
    risk_bg = colors.HexColor("#2d0a0a") if prob >= 70 else (colors.HexColor("#2d1f0a") if prob >= 40 else colors.HexColor("#0a2d12"))
    risk_border = risk_color

    center_sty = ParagraphStyle("Center", fontName="Helvetica", fontSize=9,
                                textColor=colors.HexColor("#aaaacc"), alignment=TA_CENTER)
    result_data = [
        [Paragraph(f"{diet['emoji']} {result}", risk_sty)],
        [Paragraph(f"{prob:.1f}%", percent_sty)],
        [Paragraph("Estimated Cardiac Risk Probability", center_sty)],
    ]
    result_table = Table(result_data, colWidths=[170*mm])
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), risk_bg),
        ("BOX", (0, 0), (-1, -1), 2, risk_border),
        ("ROWPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, risk_border),
    ]))
    content.append(KeepTogether([
        Paragraph("RISK ASSESSMENT RESULT", section_sty),
        HRFlowable(width="100%", thickness=0.5, color=PRIMARY),
        Spacer(1, 3*mm),
        result_table,
    ]))

    # Risk scale bar (text-based)
    content.append(Spacer(1, 3*mm))
    scale_label = Paragraph(
        f"<b>Risk Score:</b> {prob:.1f}% — "
        f"{'Seek immediate medical attention.' if prob >= 70 else 'Monitor regularly and improve lifestyle.' if prob >= 40 else 'Maintain healthy habits to stay protected.'}",
        body_sty
    )
    content.append(scale_label)
    content.append(Spacer(1, 5*mm))

    # ---- CLINICAL PARAMETERS ----
    content.append(Paragraph("CLINICAL PARAMETERS", section_sty))
    content.append(HRFlowable(width="100%", thickness=0.5, color=PRIMARY))
    content.append(Spacer(1, 3*mm))

    params = [
        ["Parameter", "Value", "Reference Range", "Status"],
        ["Blood Pressure", f"{input_data['trestbps']} mmHg", "90–120 mmHg",
         "⚠️ Elevated" if input_data['trestbps'] > 130 else "✅ Normal"],
        ["Cholesterol", f"{input_data['chol']} mg/dL", "< 200 mg/dL",
         "⚠️ High" if input_data['chol'] > 200 else "✅ Normal"],
        ["Max Heart Rate", f"{input_data['thalach']} bpm", "60–100 bpm (rest)",
         "✅ Active" if input_data['thalach'] > 100 else "⚠️ Low"],
        ["ST Depression", f"{input_data['oldpeak']}", "0.0–1.0",
         "⚠️ Elevated" if input_data['oldpeak'] > 1.0 else "✅ Normal"],
        ["Fasting Blood Sugar", "High" if input_data['fbs'] else "Normal", "< 120 mg/dL",
         "⚠️ High" if input_data['fbs'] else "✅ Normal"],
        ["Exercise Angina", "Yes" if input_data['exang'] else "No", "No (ideal)",
         "⚠️ Present" if input_data['exang'] else "✅ Absent"],
        ["Major Vessels", str(input_data['ca']), "0 (ideal)",
         f"{'⚠️ ' if input_data['ca'] > 0 else '✅ '}{input_data['ca']} vessel(s)"],
    ]

    param_table = Table(params, colWidths=[52*mm, 33*mm, 45*mm, 40*mm])
    param_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWPADDING", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d0d8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR", (0, 1), (0, -1), TEXT),
    ]))
    content.append(param_table)
    content.append(Spacer(1, 5*mm))

    # ---- DIET PLAN ----
    content.append(Paragraph("PERSONALISED DIET PLAN", section_sty))
    content.append(HRFlowable(width="100%", thickness=0.5, color=PRIMARY))
    content.append(Spacer(1, 3*mm))

    # Foods to eat
    eat_rows = [["✅ FOODS TO EAT DAILY", ""]]
    for icon, food in diet["foods_eat"]:
        eat_rows.append([Paragraph(icon, body_sty), Paragraph(food, body_sty)])

    eat_table = Table(eat_rows, colWidths=[12*mm, 158*mm])
    eat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0a3d20")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#44ff88")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("SPAN", (0, 0), (-1, 0)),
        ("ROWPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0fff5"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#c0f0d0")),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
    ]))
    content.append(eat_table)
    content.append(Spacer(1, 3*mm))

    # Foods to avoid
    avoid_rows = [["❌ FOODS TO AVOID", ""]]
    for icon, food in diet["foods_avoid"]:
        avoid_rows.append([Paragraph(icon, body_sty), Paragraph(food, body_sty)])

    avoid_table = Table(avoid_rows, colWidths=[12*mm, 158*mm])
    avoid_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3d0a0a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#ff6b6b")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("SPAN", (0, 0), (-1, 0)),
        ("ROWPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#fff5f5"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#f0c0c0")),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
    ]))
    content.append(avoid_table)
    content.append(Spacer(1, 4*mm))

    # Sample Meal Plan
    content.append(Paragraph("SUGGESTED DAILY MEAL PLAN", section_sty))
    content.append(HRFlowable(width="100%", thickness=0.5, color=PRIMARY))
    content.append(Spacer(1, 3*mm))

    meal_rows = [["Meal Time", "Suggested Food"]]
    for meal, food in diet["meals"].items():
        meal_rows.append([
            Paragraph(f"<b>{meal}</b>", bold_sty),
            Paragraph(food, body_sty)
        ])

    meal_table = Table(meal_rows, colWidths=[40*mm, 130*mm])
    meal_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWPADDING", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d0d8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_BG, colors.white]),
    ]))
    content.append(meal_table)
    content.append(Spacer(1, 5*mm))

    # ---- LIFESTYLE ----
    content.append(Paragraph("LIFESTYLE RECOMMENDATIONS", section_sty))
    content.append(HRFlowable(width="100%", thickness=0.5, color=PRIMARY))
    content.append(Spacer(1, 3*mm))

    life_rows = [[Paragraph(tip, body_sty)] for tip in diet["lifestyle"]]
    life_table = Table(life_rows, colWidths=[170*mm])
    life_table.setStyle(TableStyle([
        ("ROWPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_BG, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#d0d8f0")),
    ]))
    content.append(life_table)
    content.append(Spacer(1, 5*mm))

    # ---- DISCLAIMER ----
    disc = Paragraph(
        "<b>⚠️ Medical Disclaimer:</b> This report is generated by an AI-based predictive model for informational purposes only. "
        "It is NOT a substitute for professional medical diagnosis, advice, or treatment. "
        "Please consult a qualified cardiologist or healthcare provider before making any medical decisions.",
        ParagraphStyle("Disc", fontName="Helvetica", fontSize=8, textColor=GRAY,
                       backColor=colors.HexColor("#fff8e1"), borderPadding=8, leading=13,
                       borderColor=colors.HexColor("#f0c040"), borderWidth=0.5)
    )
    content.append(disc)
    content.append(Spacer(1, 3*mm))

    # ---- FOOTER ----
    footer_data = [[
        Paragraph("CardioAI v2.0", small_sty),
        Paragraph(f"Generated: {now}", small_sty),
        Paragraph("Confidential Medical Record", small_sty),
    ]]
    footer_table = Table(footer_data, colWidths=[56*mm, 58*mm, 56*mm])
    footer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#7a8aa0")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (0, 0), (0, 0), "LEFT"),
        ("ALIGN", (1, 0), (1, 0), "CENTER"),
        ("ALIGN", (2, 0), (2, 0), "RIGHT"),
    ]))
    content.append(footer_table)

    doc.build(content)
    return file_name


# =========================
# PREDICT BUTTON
# =========================
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🚀 Analyze Heart Risk", use_container_width=True)

if predict_clicked:
    if not name.strip():
        st.error("Please enter the patient's name.")
    else:
        input_data = {
            "trestbps": trestbps, "chol": chol, "thalach": thalach,
            "oldpeak": oldpeak, "fbs": fbs, "restecg": restecg,
            "exang": exang, "slope": slope, "ca": ca
        }

        data_array = np.array([[age, sex, cp, trestbps, chol,
                                 fbs, restecg, thalach,
                                 exang, oldpeak, slope, ca, thal]])

        if model_loaded:
            prob_raw = model.predict_proba(data_array)[0][1]
        else:
            # Demo: random for testing
            prob_raw = np.random.uniform(0.1, 0.95)

        prob_pct = prob_raw * 100

        if prob_pct >= 70:
            result = "HIGH RISK"
            result_class = "result-high"
            result_emoji = "🔴"
        elif prob_pct >= 65:
            result = "MODERATE RISK"
            result_class = "result-moderate"
            result_emoji = "🟡"
        else:
            result = "LOW RISK"
            result_class = "result-low"
            result_emoji = "🟢"

        diet = get_diet_plan(prob_pct)

        st.markdown("<br>", unsafe_allow_html=True)

        # Result display
        st.markdown(f"""
        <div class="{result_class}">
            <p class="result-title">{result_emoji} {result}</p>
            <p class="risk-percent">{prob_pct:.1f}%</p>
            <p style="color: rgba(255,255,255,0.6); margin:0; font-size:0.9rem;">Estimated Cardiac Risk Probability</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(min(prob_pct / 100, 1.0))

        # Diet section
        st.markdown("### 🥗 Your Personalised Diet Plan")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**✅ Eat These Daily**")
            for icon, food in diet["foods_eat"]:
                st.markdown(f'<div class="diet-card"><span class="diet-icon">{icon}</span>{food}</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown("**❌ Avoid These**")
            for icon, food in diet["foods_avoid"]:
                st.markdown(f'<div class="diet-card"><span class="diet-icon">{icon}</span>{food}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🍽️ Suggested Daily Meal Plan**")
        for meal, food in diet["meals"].items():
            st.markdown(f'<div class="diet-card"><b>{meal}:</b> {food}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**💡 Lifestyle Tips**")
        cols = st.columns(2)
        for i, tip in enumerate(diet["lifestyle"]):
            with cols[i % 2]:
                st.markdown(f'<div class="diet-card">{tip}</div>', unsafe_allow_html=True)

        # PDF Download
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_file = generate_pdf(name, age, gender, doctor, prob_pct, result, diet, input_data)

        with open(pdf_file, "rb") as f:
            st.download_button(
                label="📄 Download Professional Report (PDF)",
                data=f,
                file_name=f"CardioAI_Report_{name.replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    CardioAI v2.0 &nbsp;•&nbsp; For educational & research purposes only &nbsp;•&nbsp; Not a substitute for medical advice<br>
    Always consult a qualified cardiologist for diagnosis and treatment.
</div>
""", unsafe_allow_html=True)