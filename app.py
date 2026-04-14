import streamlit as st
import pandas as pd
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Web UI Setup)
# ==========================================
st.set_page_config(
    page_title="NLP Insight for SME", 
    page_icon="💡", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 🎨 ตกแต่งด้วย Custom CSS (โทนสีส้ม) ---
st.markdown("""
    <style>
    /* ปรับแต่งหัวข้อหลัก */
    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #E65100; /* สีส้มเข้ม */
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 18px;
        color: #616161;
        text-align: center;
        margin-bottom: 30px;
    }
    /* ปรับแต่งกล่องผลลัพธ์ */
    .result-box {
        background-color: #FFF3E0; /* พื้นหลังสีส้มอ่อนๆ พาสเทล */
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF9800; /* แถบขอบสีส้มสด */
        margin-top: 20px;
    }
    /* ปรับแต่งช่องกรอกข้อความ */
    .stTextArea textarea {
        font-size: 16px !important;
        border-radius: 10px !important;
        border: 1px solid #FFB74D !important; /* กรอบช่องกรอกข้อความสีส้ม */
    }
    .stTextArea textarea:focus {
        border-color: #FF9800 !important;
        box-shadow: 0 0 0 0.2rem rgba(255, 152, 0, 0.25) !important;
    }
    /* ปรับแต่งปุ่มกดให้เป็นสีส้ม */
    div.stButton > button:first-child {
        background-color: #FF9800 !important; /* สีปุ่มส้ม */
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #F57C00 !important; /* สีส้มเข้มขึ้นเมื่อเอาเมาส์ชี้ */
        box-shadow: 0px 4px 10px rgba(245, 124, 0, 0.3) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 📱 แถบเมนูด้านข้าง (Sidebar)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3126/3126400.png", width=80) 
    st.title("📌 ข้อมูลโครงงาน")
    st.info("**หัวข้อ:** NLP Insight for SME\n\nการวิเคราะห์ข้อมูลเชิงลึกด้วย NLP เพื่อ SME")
    
    st.markdown("---")
    st.markdown("### 💡 ทำไมต้องใช้ระบบนี้?")
    st.markdown("""
    - ช่วยอ่านรีวิวแทนเจ้าของร้าน
    - แยกแยะอารมณ์ลูกค้า (บวก/ลบ/ผสม)
    - จับประเด็นปัญหาได้อย่างแม่นยำ
    - ให้คำแนะนำเชิงธุรกิจอัตโนมัติ
    """)
    
    st.markdown("---")
    st.caption("พัฒนาด้วย 💻 Python, Scikit-learn, PyThaiNLP และ Streamlit")

# ==========================================
# 🎯 ส่วนหัวของหน้าจอหลัก
# ==========================================
st.markdown('<p class="main-title">✨ NLP Insight for SME</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">ระบบวิเคราะห์รีวิวลูกค้าและสร้างคำแนะนำเชิงธุรกิจอัตโนมัติด้วย AI</p>', unsafe_allow_html=True)

# ==========================================
# 2. เตรียมระบบ AI (Backend Logic - Mega Dataset)
# ==========================================
custom_stopwords = set(thai_stopwords()) - {"ไม่", "แต่", "ดี", "แย่", "น้อย", "แพง", "ช้า", "นาน", "ไว", "ห่วย", "แดก", "หมา", "โคตร", "สุด", "ค่อย", "ให้", "เยอะ", "ถูก", "จึ้ง", "นัว", "แรง", "ฉ่ำ"}

def preprocess(text):
    tokens = word_tokenize(text, engine="newmm")
    clean = [w for w in tokens if w not in custom_stopwords and w.strip()]
    return clean

@st.cache_resource 
def load_models():
    train_data = [
        # อาหาร
        ("อาหารอร่อย รสชาติดี อาหารสดใหม่", ["คุณภาพอาหาร"], "Positive"),
        ("อร่อยมาก โคตรอร่อย แสงออกปาก อร่อยแสงออกปาก", ["คุณภาพอาหาร"], "Positive"),
        ("รสชาติถูกปาก วัตถุดิบพรีเมียม ทำถึงมาก", ["คุณภาพอาหาร"], "Positive"),
        ("แซ่บมาก นัวสุดๆ รสชาติกลมกล่อม", ["คุณภาพอาหาร"], "Positive"),
        ("หมูนุ่มมาก ละลายในปาก ไม่คาวเลย", ["คุณภาพอาหาร"], "Positive"),
        ("ตรงปกไม่จกตา รสชาติเหมือนกินที่บ้าน", ["คุณภาพอาหาร"], "Positive"),
        ("อร่อยจนต้องซ้ำ สั่งกินทุกวันเลย", ["คุณภาพอาหาร"], "Positive"),
        ("น้ำซุปเข้มข้น หอมเครื่องเทศ", ["คุณภาพอาหาร"], "Positive"),
        ("ไม่อร่อยเลย รสชาติแย่ อาหารจืดชืด", ["คุณภาพอาหาร"], "Negative"),
        ("รสชาติหมาไม่แดก แดกไม่ได้เลย จืดสนิท", ["คุณภาพอาหาร"], "Negative"),
        ("เค็มปี๋ หวานตัดขา เปรี้ยวจี๊ด เค็มไตพัง", ["คุณภาพอาหาร"], "Negative"),
        ("อาหารบูด มีกลิ่นเหม็น ไม่สด", ["คุณภาพอาหาร"], "Negative"),
        ("หมูเหนียวมาก เคี้ยวจนกรามค้าง เหนียวเป็นยางหนังสติ๊ก", ["คุณภาพอาหาร"], "Negative"),
        ("กินแล้วท้องเสีย ท้องร่วง อาหารเป็นพิษ", ["คุณภาพอาหาร"], "Negative"),
        ("มีแมลงสาบในอาหาร เจอเส้นผม มีหนอน", ["คุณภาพอาหาร", "สภาพแวดล้อมร้าน"], "Negative"),
        ("ข้าวแข็งเป็นหิน อาหารไม่สุก", ["คุณภาพอาหาร"], "Negative"),
        ("รสชาติแปลกๆ เหมือนของค้างคืน", ["คุณภาพอาหาร"], "Negative"),

        # บริการ
        ("พนักงานบริการดี พนักงานน่ารัก พูดจาสุภาพ ดูแลเอาใจใส่มาก", ["การบริการพนักงาน"], "Positive"),
        ("เทคแคร์ลูกค้าดีมาก ยิ้มแย้มแจ่มใส", ["การบริการพนักงาน"], "Positive"),
        ("เจ้าของร้านใจดี เป็นกันเอง แนะนำเมนูดีมาก", ["การบริการพนักงาน"], "Positive"),
        ("เปลี่ยนจานให้ตลอด บริการระดับ 5 ดาว", ["การบริการพนักงาน"], "Positive"),
        ("พี่พนักงานพูดจาเพราะมาก บริการประทับใจ", ["การบริการพนักงาน"], "Positive"),
        ("ใส่ใจรายละเอียด จำเมนูประจำได้ด้วย", ["การบริการพนักงาน"], "Positive"),
        ("พนักงานพูดจาไม่สุภาพ บริการแย่ เรียกพนักงานยาก", ["การบริการพนักงาน"], "Negative"),
        ("พนักงานหน้าเป็นตูด ชักสีหน้าใส่ เหวี่ยงลูกค้า", ["การบริการพนักงาน"], "Negative"),
        ("บริการห่วยแตก ไม่สนใจลูกค้า มารยาททราม", ["การบริการพนักงาน"], "Negative"),
        ("บริการไม่ค่อยดี มารยาทแย่ บริการห่วย พนักพูดจาแย่มาก", ["การบริการพนักงาน"], "Negative"),
        ("พูดจาหมาไม่แดก ด่าลูกค้า ไล่ลูกค้า", ["การบริการพนักงาน"], "Negative"),
        ("พนักงานเอาแต่เล่นมือถือ เม้าท์แตก ไม่รับออเดอร์", ["การบริการพนักงาน"], "Negative"),
        ("แม่ค้าปากแจ๋ว ด่าไรเดอร์ ด่าลูกค้า", ["การบริการพนักงาน"], "Negative"),

        # ความเร็ว
        ("ได้อาหารเร็ว เสิร์ฟไว ไม่ต้องรอนาน", ["ความเร็วในการให้บริการ"], "Positive"),
        ("ทำไวมาก แป๊บเดียวได้กิน เสิร์ฟเร็วปานสายฟ้า", ["ความเร็วในการให้บริการ"], "Positive"),
        ("จัดการคิวดีมาก รันคิวไว ไม่ต้องรอคิวนาน", ["ความเร็วในการให้บริการ"], "Positive"),
        ("สั่งปุ๊บได้ปั๊บ อาหารออกต่อเนื่องดีมาก", ["ความเร็วในการให้บริการ"], "Positive"),
        ("รออาหารนาน รอนาน อาหารออกช้า", ["ความเร็วในการให้บริการ"], "Negative"),
        ("รอจนเหงือกแห้ง ช้าโคตรๆ ลืมออเดอร์", ["ความเร็วในการให้บริการ"], "Negative"),
        ("ตามอาหารสามรอบก็ยังไม่ได้ โต๊ะมาทีหลังได้ก่อน", ["ความเร็วในการให้บริการ"], "Negative"),
        ("หิวจนหายหิว รอเป็นชั่วโมง", ["ความเร็วในการให้บริการ"], "Negative"),
        ("ทำผิดคิว ข้ามคิว ระบบจัดการแย่มาก", ["ความเร็วในการให้บริการ"], "Negative"),
        ("ไรเดอร์รอนานมาก ร้านทำช้าสุดๆ", ["ความเร็วในการให้บริการ"], "Negative"),

        # ราคา
        ("ราคาคุ้มค่า ราคาเหมาะสม ราคาถูก ให้เยอะ", ["ราคา"], "Positive"),
        ("ถูกและดี ไม่แพงเลย สบายกระเป๋า คุ้มเงินสุดๆ", ["ราคา"], "Positive"),
        ("ให้เยอะจุกๆ อิ่มจนจุก คุ้มราคามาก ราคาไม่แพง", ["ราคา"], "Positive"),
        ("สมราคา วัตถุดิบสมกับราคาที่จ่าย", ["ราคา"], "Positive"),
        ("จัดโปรบ่อย ลดราคาเยอะ คุ้มมาก", ["ราคา"], "Positive"),
        ("ราคาแพง ราคาแพงไปนิด", ["ราคา"], "Negative"),
        ("แพงหูฉี่ ขูดรีด แพงเกินเบอร์", ["ราคา"], "Negative"),
        ("ให้น้อยมาก เหมือนไหว้เจ้า วิญญาณหมู ไม่สมราคา", ["ราคา"], "Negative"),
        ("บวก Service Charge โหดมาก แอบแฝง", ["ราคา"], "Negative"),
        ("ราคาในเมนูกับตอนคิดเงินไม่ตรงกัน โกงราคา", ["ราคา"], "Negative"),
        ("แพงแล้วยังไม่อร่อย เสียดายเงิน", ["ราคา", "คุณภาพอาหาร"], "Negative"),

        # สภาพแวดล้อม
        ("บรรยากาศดี ร้านสะอาด แอร์เย็นสบาย ร้านกว้างขวางดี", ["สภาพแวดล้อมร้าน"], "Positive"),
        ("ห้องน้ำสะอาด ที่จอดรถกว้างขวาง", ["สภาพแวดล้อมร้าน"], "Positive"),
        ("ร้านสวยมาก ถ่ายรูปสวย มุมถ่ายรูปเยอะ", ["สภาพแวดล้อมร้าน"], "Positive"),
        ("แอร์เย็นฉ่ำ เพลงเพราะ นั่งชิลดี", ["สภาพแวดล้อมร้าน"], "Positive"),
        ("มีที่จอดรถเยอะมาก สะดวกสบาย", ["สภาพแวดล้อมร้าน"], "Positive"),
        ("โต๊ะสกปรก ร้านดูแคบ แอร์ไม่เย็น", ["สภาพแวดล้อมร้าน"], "Negative"),
        ("สกปรกมาก มีแมลงสาบ หนูวิ่งพล่าน ห้องน้ำเหม็น", ["สภาพแวดล้อมร้าน"], "Negative"),
        ("แมลงวันเยอะมาก ปัดจนไม่ได้กินข้าว", ["สภาพแวดล้อมร้าน"], "Negative"),
        ("ร้านร้อนมาก เหม็นอับ โต๊ะเหนียว", ["สภาพแวดล้อมร้าน"], "Negative"),
        ("ที่จอดรถหายาก ไม่มีที่จอดรถ ซอยแคบ", ["สภาพแวดล้อมร้าน"], "Negative"),
        ("เสียงดังโวยวาย หนวกหู นั่งไม่สบาย", ["สภาพแวดล้อมร้าน"], "Negative"),

        # ผสม
        ("ร้านนี้อาหารอร่อยครับให้เยอะราคาไม่แพงเสียอย่างเดียวบริการไม่ค่อยดี", ["คุณภาพอาหาร", "ราคา", "การบริการพนักงาน"], "Mixed"),
        ("ร้านกว้างขวางดี แอร์เย็นฉ่ำ น้องๆ พนักงานดูแลเอาใจใส่มาก คุ้มเงินสุดๆ", ["สภาพแวดล้อมร้าน", "การบริการพนักงาน", "ราคา"], "Positive"),
        ("ถูกและดี แต่แมลงวันเยอะไปหน่อย", ["ราคา", "สภาพแวดล้อมร้าน"], "Mixed"),
        ("อาหารอร่อยนะ แต่รอนานไปหน่อย หิวจนตาลาย", ["คุณภาพอาหาร", "ความเร็วในการให้บริการ"], "Mixed"),
        ("แอร์เย็น ร้านสวย แต่กาแฟจืดชืดมาก แถมราคาแพงหูฉี่", ["สภาพแวดล้อมร้าน", "คุณภาพอาหาร", "ราคา"], "Mixed"),
        ("พนักงานบริการดีมาก ยิ้มแย้ม แต่หมูเหนียวเคี้ยวไม่ขาด", ["การบริการพนักงาน", "คุณภาพอาหาร"], "Mixed"),
        ("ได้เยอะ คุ้มราคา แต่หาที่จอดรถยากมาก ต้องจอดแปะริมถนน", ["ราคา", "สภาพแวดล้อมร้าน"], "Mixed"),
        ("แพงแถมยังไม่อร่อย พนักงานก็หน้าบูดอีก ลาขาดร้านนี้", ["ราคา", "คุณภาพอาหาร", "การบริการพนักงาน"], "Negative")
    ]

    X_train_raw = [" ".join(preprocess(text)) for text, _, _ in train_data]
    y_aspect_train = [aspects for _, aspects, _ in train_data]
    y_sentiment_train = [sentiment for _, _, sentiment in train_data]

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), token_pattern=None, ngram_range=(1, 3))
    X_train_vec = vectorizer.fit_transform(X_train_raw)

    mlb = MultiLabelBinarizer()
    y_aspect_bin = mlb.fit_transform(y_aspect_train)

    topic_model = OneVsRestClassifier(RandomForestClassifier(n_estimators=150, random_state=42))
    topic_model.fit(X_train_vec, y_aspect_bin)

    sentiment_model = RandomForestClassifier(n_estimators=150, random_state=42)
    sentiment_model.fit(X_train_vec, y_sentiment_train)

    return vectorizer, topic_model, sentiment_model, mlb

vectorizer, topic_model, sentiment_model, mlb = load_models()

# ==========================================
# 3. ฟังก์ชันสร้างคำแนะนำ 
# ==========================================
def generate_recommendation(aspects, sentiment):
    suggestions = []
    if sentiment in ["Negative", "Mixed"]:
        if "ความเร็วในการให้บริการ" in aspects: suggestions.append("⚠️ ควรปรับปรุงความเร็วในการให้บริการ หรือจัดการระบบคิว/ออเดอร์ใหม่")
        if "การบริการพนักงาน" in aspects: suggestions.append("⚠️ ควรจัดอบรมพนักงานด้านการบริการ มารยาท และการสื่อสารกับลูกค้าด่วน")
        if "คุณภาพอาหาร" in aspects: suggestions.append("⚠️ ควรตรวจสอบสูตรอาหาร ความสะอาดของวัตถุดิบ และควบคุมรสชาติให้ได้มาตรฐาน")
        if "ราคา" in aspects: suggestions.append("⚠️ ควรพิจารณาราคาให้สอดคล้องกับปริมาณ หรือทำโปรโมชั่นเพิ่มเติม")
        if "สภาพแวดล้อมร้าน" in aspects: suggestions.append("⚠️ ควรดูแลความสะอาด จัดการพื้นที่ร้าน และเช็คระบบปรับอากาศ")
        
    if sentiment in ["Positive", "Mixed"]:
        if "ความเร็วในการให้บริการ" in aspects: suggestions.append("✅ รักษามาตรฐานความรวดเร็วในการเสิร์ฟต่อไป")
        if "การบริการพนักงาน" in aspects: suggestions.append("✅ รักษามาตรฐานการบริการที่ดีเยี่ยมนี้ไว้")
        if "คุณภาพอาหาร" in aspects: suggestions.append("✅ รักษาคุณภาพและรสชาติอาหารที่ยอดเยี่ยมนี้ไว้")
        if "ราคา" in aspects: suggestions.append("✅ รักษาราคาที่คุ้มค่าและปริมาณที่เหมาะสมต่อไป")
        if "สภาพแวดล้อมร้าน" in aspects: suggestions.append("✅ รักษาสภาพแวดล้อมร้านและความสะอาดเพื่อดึงดูดลูกค้า")
        
    return suggestions if suggestions else ["📌 บันทึกความคิดเห็นเพื่อนำไปวิเคราะห์ภาพรวมต่อไป"]

# ==========================================
# 4. ส่วนรับข้อมูลจากผู้ใช้งาน (User Input)
# ==========================================

# 💡 เมนูพับเก็บได้ สำหรับตัวอย่างรีวิว
with st.expander("📋 คลิกดูตัวอย่างข้อความรีวิวสำหรับทดสอบ"):
    st.markdown("""
    - `อาหารอร่อยมาก รสชาติดี แต่รออาหารนานไปหน่อย` (Mixed)
    - `ร้านกว้างขวางดี แอร์เย็นฉ่ำ น้องๆ พนักงานดูแลเอาใจใส่มาก คุ้มเงินสุดๆ` (Positive)
    - `แพงแถมยังไม่อร่อย พนักงานก็หน้าบูดอีก ลาขาดร้านนี้` (Negative)
    - `ร้านนี้อาหารอร่อยครับให้เยอะราคาไม่แพงเสียอย่างเดียวบริการไม่ค่อยดี` (Mixed แบบไม่เว้นวรรค)
    """)

# กล่องกรอกข้อมูลหลัก
user_input = st.text_area(
    "✍️ กรอกข้อความรีวิวจากลูกค้า:", 
    height=120, 
    placeholder="พิมพ์ข้อความรีวิวที่ต้องการวิเคราะห์ที่นี่..."
)

# ปุ่มกดที่อยู่ตรงกลาง
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    # ปุ่มจะถูกควบคุมสีส้มจาก CSS ด้านบน
    submit_btn = st.button("🔍 เริ่มวิเคราะห์รีวิว", use_container_width=True)

if submit_btn:
    if user_input.strip() == "":
        st.warning("⚠️ กรุณากรอกข้อความรีวิวก่อนครับ")
    else:
        with st.spinner("🤖 AI กำลังประมวลผล..."):
            
            # --- HYBRID FALLBACK ---
            pos_aspects = set()
            neg_aspects = set()
            text_lower = user_input.replace(" ", "") 
            
            if any(w in text_lower for w in ["ไม่อร่อย", "หมาไม่แดก", "แดกไม่ได้", "แย่", "เค็ม", "บูด", "แมลงสาบ", "หนอน", "เส้นผม", "ท้องเสีย", "แข็ง"]):
                neg_aspects.add("คุณภาพอาหาร")
            elif any(w in text_lower for w in ["อร่อย", "สดใหม่", "รสชาติดี", "แซ่บ", "แสงออกปาก", "นัว", "เข้มข้น"]):
                pos_aspects.add("คุณภาพอาหาร")
                
            if any(w in text_lower for w in ["บริการไม่ดี", "บริการไม่ค่อยดี", "มารยาทไม่ดี", "มารยาทแย่", "หน้าเป็นตูด", "ชักสีหน้า", "เหวี่ยง", "บริการแย่", "พนักพูดจาแย่", "มารยาททราม", "ไล่ลูกค้า"]):
                neg_aspects.add("การบริการพนักงาน")
            elif any(w in text_lower for w in ["บริการดี", "สุภาพ", "น่ารัก", "เอาใจใส่", "ใจดี", "เทคแคร์", "ยิ้มแย้ม"]):
                pos_aspects.add("การบริการพนักงาน")
                
            if any(w in text_lower for w in ["แพง", "ขูดรีด", "ให้น้อย", "ไม่สมราคา", "วิญญาณหมู", "โกงราคา"]):
                neg_aspects.add("ราคา")
            elif any(w in text_lower for w in ["ไม่แพง", "ราคาถูก", "ให้เยอะ", "คุ้ม"]):
                pos_aspects.add("ราคา")
                
            if any(w in text_lower for w in ["ช้า", "รอนาน", "เหงือกแห้ง", "ลืมออเดอร์", "ข้ามคิว"]):
                neg_aspects.add("ความเร็วในการให้บริการ")
            elif any(w in text_lower for w in ["ไว", "เร็ว", "ไม่ต้องรอ", "ได้ปั๊บ"]):
                pos_aspects.add("ความเร็วในการให้บริการ")

            if any(w in text_lower for w in ["สกปรก", "แคบ", "ร้อน", "เหม็น", "หนู", "เสียงดัง"]):
                neg_aspects.add("สภาพแวดล้อมร้าน")
            elif any(w in text_lower for w in ["บรรยากาศดี", "สะอาด", "แอร์เย็น", "กว้างขวาง", "ถ่ายรูปสวย", "ที่จอดรถ"]):
                pos_aspects.add("สภาพแวดล้อมร้าน")

            # กระบวนการ ML
            processed_text = " ".join(preprocess(user_input))
            X_test_vec = vectorizer.transform([processed_text])
            
            pred_aspect_bin = topic_model.predict(X_test_vec)
            ml_aspects = list(mlb.inverse_transform(pred_aspect_bin)[0]) if X_test_vec.sum() > 0 and mlb.inverse_transform(pred_aspect_bin)[0] else []
            
            final_aspects = list(pos_aspects.union(neg_aspects).union(set(ml_aspects)))
            if not final_aspects: final_aspects = ["ทั่วไป"]
            
            if len(pos_aspects) > 0 and len(neg_aspects) > 0:
                final_sentiment = "Mixed"
            elif len(neg_aspects) > 0:
                final_sentiment = "Negative"
            elif len(pos_aspects) > 0:
                final_sentiment = "Positive"
            else:
                final_sentiment = sentiment_model.predict(X_test_vec)[0] if X_test_vec.sum() > 0 else "Neutral"
                if final_sentiment == "Mixed":
                    final_sentiment = "Negative" if "ไม่" in processed_text or "แย่" in processed_text else "Positive"

            recommendations = generate_recommendation(final_aspects, final_sentiment)

        # ==========================================
        # 5. แสดงผลลัพธ์ (Display Results - Orange Theme UI)
        # ==========================================
        st.divider()
        st.subheader("📊 ผลการวิเคราะห์ (Analysis Results)")
        
        with st.container(border=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**อารมณ์ของรีวิว (Sentiment):**")
                if final_sentiment == "Positive":
                    st.success("🟢 เชิงบวก (Positive)")
                elif final_sentiment == "Negative":
                    st.error("🔴 เชิงลบ (Negative)")
                elif final_sentiment == "Mixed":
                    st.warning("🟡 ผสม (Mixed - มีทั้งชมและติ)") 
                else:
                    st.info("⚪ เป็นกลาง (Neutral)")
                    
            with col2:
                st.markdown("**ประเด็นที่พูดถึง (Aspects):**")
                aspect_tags = " | ".join([f"🏷️ {a}" for a in final_aspects])
                st.warning(aspect_tags) # ใช้ st.warning เพื่อให้ได้กรอบสีส้ม/เหลืองอ่อนเข้ากับธีม

        # กล่องคำแนะนำสีส้มพาสเทล
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("#### 💡 คำแนะนำเชิงธุรกิจ (Actionable Insights)")
        for rec in recommendations:
            st.markdown(f"- {rec}")
        st.markdown('</div>', unsafe_allow_html=True)