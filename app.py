import os
import streamlit as st
import requests
import json
import PyPDF2
import base64
import html
from dotenv import load_dotenv

# --- 🔑 CORE CONFIGURATION ---
load_dotenv()
API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_GENAI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or ""
).strip()

# --- 📚 CURRICULUM DATABASE ---
CURRICULUM = {
    "9th": {
        "Physics": ["Motion & Force", "Work & Energy", "Waves", "Sound"],
        "Chemistry": ["Matter States", "Atomic Structure", "Chemical Bonds", "Reactions"],
        "Biology": ["Cell Structure", "Photosynthesis", "Respiration", "Plant Tissues"],
    },
    "10th": {
        "Physics": ["Electricity", "Magnetism", "Light", "Modern Physics Intro"],
        "Chemistry": ["Periodic Table", "Acids & Bases", "Salts", "Carbon & Compounds"],
        "Biology": ["Digestion", "Circulation", "Nervous System", "Reproduction"],
    },
    "11th": {
        "Physics": ["Mechanics", "Thermodynamics", "Optics", "Waves & Oscillations"],
        "Chemistry": ["Organic Compounds", "Chemical Kinetics", "Equilibrium", "Redox"],
        "Biology": ["Photosynthesis Deep", "Genetics", "Evolution", "Ecology"],
    },
    "12th": {
        "Physics": ["Relativity", "Quantum Mechanics", "Nuclear Physics", "Semiconductors"],
        "Chemistry": ["Electrochemistry", "Coordination Compounds", "Polymers", "Environmental"],
        "Biology": ["Molecular Biology", "Biotechnology", "Immune System", "Homeostasis"],
    },
}

# --- 📋 MODEL DISCOVERY ---
def get_available_models():
    if not API_KEY:
        return []

    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        response = requests.get(url, headers={"x-goog-api-key": API_KEY}, timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [
                m["name"].split("/")[-1]
                for m in models
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
        return []
    except Exception:
        return []

AVAILABLE_MODELS = get_available_models()
PREFERRED_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"]
SELECTED_MODEL = None

for model in PREFERRED_MODELS:
    if model in AVAILABLE_MODELS or not AVAILABLE_MODELS:
        SELECTED_MODEL = model
        break

if not SELECTED_MODEL:
    SELECTED_MODEL = PREFERRED_MODELS[0]

# --- 🖼️ IMAGE PROCESSING ---
def encode_image(image_file):
    return {
        "mime_type": image_file.type or "image/jpeg",
        "data": base64.b64encode(image_file.read()).decode(),
    }

def extract_pdf_text(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        pages = []
        for page in reader.pages[:5]:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text.strip())
        return "\n\n".join(pages)
    except Exception:
        return ""

def format_chat_text(text):
    return html.escape(text).replace("\n", "<br>")

# --- 🧠 AI RESPONSE ENGINE ---
def get_concept_explanation(prompt, grade, subject, topic, language, image_b64=None):
    """Generate concept explanation with language support"""
    if not API_KEY:
        raise Exception("Missing API key. Add GEMINI_API_KEY to your .env file.")
    
    lang_instructions = {
        "Roman Urdu": "Answer in Roman Urdu (Hinglish). Use simple, everyday examples from Pakistan.",
        "English": "Answer in clear English. Use simple examples suitable for a high school student.",
        "Urdu": "جواب اردو میں دیں۔ سادہ اردو میں سمجھائیں۔"
    }

    system_prompt = f"""You are an expert science tutor for {grade}th grade {subject} in Pakistan schools.

Your task:
- Explain concepts clearly using local Pakistani examples
- Break complex ideas into simple steps
- Use real-life analogies students can relate to
- Avoid rote memorization; focus on understanding
- For {topic}: provide a complete, step-by-step explanation
- Include definition, key points, 2-3 examples, and real-life applications
- If the question includes an image, explain the visible problem or diagram directly

{lang_instructions.get(language, lang_instructions['Roman Urdu'])}

Use clear headings and short bullet points. Keep the answer complete but easy for a high school student."""

    parts = [{"text": f"{system_prompt}\n\nQuestion: {prompt}"}]
    if image_b64:
        parts.insert(0, {"inline_data": image_b64})

    models = [SELECTED_MODEL] if SELECTED_MODEL else []
    models.extend(model for model in PREFERRED_MODELS if model not in models)
    if AVAILABLE_MODELS:
        models = [model for model in models if model in AVAILABLE_MODELS]

    last_error = "No response"
    for model in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            payload = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1536,
                },
            }
            if model.startswith("gemini-2.5"):
                payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}

            response = requests.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": API_KEY,
                },
                timeout=30,
            )
            if response.status_code == 200:
                result = response.json()
                parts_out = result.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text = "".join(part.get("text", "") for part in parts_out).strip()
                finish_reason = result.get("candidates", [{}])[0].get("finishReason", "")
                if finish_reason == "MAX_TOKENS":
                    last_error = f"{model} response was cut off by token limit"
                    continue
                if text:
                    return text
                last_error = "Empty response"
            else:
                try:
                    detail = response.json().get("error", {}).get("message", response.text)
                except ValueError:
                    detail = response.text
                last_error = f"HTTP {response.status_code}: {detail[:180]}"
        except Exception as e:
            last_error = str(e)[:180]

    raise Exception(f"API Error - {last_error}")

# --- 🎨 UI SETUP ---
st.set_page_config(page_title="Ustaad AI - Science Tutor", page_icon="🧑‍🎓", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; background: #08101f; color: #e2e8f0; }
    .stApp { background: linear-gradient(180deg, #06101c 0%, #0e1f3f 100%); }
    .chat-container { background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(96,165,250,0.14); border-radius: 24px; padding: 1.4rem; min-height: 400px; max-height: 600px; overflow-y: auto; }
    .msg-user { background: linear-gradient(135deg, rgba(248,113,113,0.2) 0%, rgba(248,113,113,0.1) 100%); border-left: 4px solid #f87171; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
    .msg-assistant { background: linear-gradient(135deg, rgba(96,165,250,0.2) 0%, rgba(96,165,250,0.1) 100%); border-left: 4px solid #60a5fa; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
    .msg-user strong { color: #f87171; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .msg-assistant strong { color: #60a5fa; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .topic-badge { background: rgba(96,165,250,0.2); border: 1px solid rgba(96,165,250,0.3); border-radius: 20px; padding: 0.5rem 1rem; display: inline-block; margin-right: 0.5rem; margin-bottom: 0.5rem; }
    .info-card { background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.2); border-radius: 12px; padding: 1rem; margin-bottom: 1rem; text-align: center; }
    .small-muted { color: #94a3b8; font-size: 0.9rem; }
    .input-section { background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(96,165,250,0.1); border-radius: 16px; padding: 1.2rem; margin-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_grade" not in st.session_state:
    st.session_state.selected_grade = "9th"
if "selected_subject" not in st.session_state:
    st.session_state.selected_subject = "Physics"
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "Roman Urdu"

# --- SIDEBAR: CURRICULUM NAVIGATION ---
with st.sidebar:
    st.markdown("<h1 style='color:#60a5fa; margin-bottom: 0.5rem;'>🧑‍🎓 USTAAD <span style='color:#fb7185;'>AI</span></h1>", unsafe_allow_html=True)
    st.markdown("<p class='small-muted'>Science Tutor for Grades 9-12</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📖 Select Your Level")
    grade = st.selectbox("Grade", ["9th", "10th", "11th", "12th"], key="grade_select")
    st.session_state.selected_grade = grade

    st.markdown("### 🔬 Choose Subject")
    subjects = list(CURRICULUM[grade].keys())
    subject = st.selectbox("Subject", subjects, key="subject_select")
    st.session_state.selected_subject = subject

    st.markdown("### 📚 Select Topic")
    topics = CURRICULUM[grade][subject]
    topic = st.selectbox("Topic", topics, key="topic_select")
    st.session_state.selected_topic = topic

    st.divider()
    st.markdown("### 🗣️ Response Language")
    language = st.selectbox("Language", ["Roman Urdu", "English", "Urdu"], key="lang_select")
    st.session_state.selected_language = language

    st.divider()
    st.markdown(f"<p class='small-muted'><strong>Model:</strong> {SELECTED_MODEL}</p>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CONTENT AREA ---
st.markdown(
    f"""
    <div style='display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-bottom:1.5rem; padding-bottom:1rem; border-bottom:1px solid rgba(96,165,250,0.1);'>
        <div>
            <h1 style='margin:0; color:#ffffff; font-size:1.8rem;'>📚 Concept Explained</h1>
            <p style='margin:0.25rem 0 0 0; color:#94a3b8; font-size:0.95rem;'>{st.session_state.selected_grade} • {st.session_state.selected_subject}</p>
        </div>
        <div style='padding:0.75rem 1.25rem; background:linear-gradient(135deg, rgba(96,165,250,0.2), rgba(96,165,250,0.05)); border:1px solid rgba(96,165,250,0.3); border-radius:18px;'>
            <span style='color:#60a5fa; font-weight:600; font-size:0.95rem;'>{st.session_state.selected_topic}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

# Display chat history
if st.session_state.messages:
    for msg in st.session_state.messages:
        content = format_chat_text(msg["content"])
        if msg["role"] == "user":
            st.markdown(f"<div class='msg-user'><strong>👤 You:</strong><br><p style='margin:0.5rem 0 0 0; line-height:1.6;'>{content}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='msg-assistant'><strong>🧑‍🎓 Ustaad AI:</strong><br><p style='margin:0.5rem 0 0 0; line-height:1.6;'>{content}</p></div>", unsafe_allow_html=True)
else:
    st.markdown(
        "<div class='info-card'><h3 style='margin:0; color:#22c55e;'>👋 Welcome to Ustaad AI!</h3><p style='margin:0.5rem 0 0 0; color:#94a3b8;'>Select a grade, subject, and topic from the sidebar 👈<br>Then ask your question and I'll explain it clearly with examples.</p></div>",
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# --- INPUT SECTION ---
st.markdown("### 📝 How would you like to learn?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Type a Question**")
    text_query = st.text_area("Write your question here:", placeholder="Ask about the concept...", height=100, key="text_input")

with col2:
    st.markdown("**Upload Image**")
    image_file = st.file_uploader("Problem image or diagram:", type=["jpg", "jpeg", "png"], key="img_upload")

with col3:
    st.markdown("**Paste Passage**")
    passage = st.text_area("Paste a passage to explain:", placeholder="Copy-paste text from your book or notes...", height=100, key="passage_input")

st.divider()

# --- VOICE INPUT (Coming Soon) ---
with st.expander("🎙️ Voice Input (Coming Soon)"):
    st.info("Record your question in Urdu, English, or Roman Urdu. Transcription will be added in the next update.")

# --- PROCESS QUERY ---
st.markdown("### Send Your Query")
send_button = st.button("Ask Ustaad 🚀", use_container_width=True)

if send_button:
    if not st.session_state.selected_topic:
        st.error("Please select a topic from the sidebar first.")
    else:
        # Determine what was input
        query_content = text_query if text_query else (passage if passage else "Please explain this concept with examples and real-life applications.")
        image_b64 = None
        
        if image_file:
            image_b64 = encode_image(image_file)
            query_content = f"I have an image of a problem. {query_content if query_content != 'Please explain this concept with examples and real-life applications.' else 'Can you explain this?'}"

        if not query_content.strip():
            st.warning("Please enter a question or select an image.")
        else:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": query_content[:200] + "..." if len(query_content) > 200 else query_content})

            with st.spinner("Ustaad is thinking..."):
                try:
                    answer = get_concept_explanation(
                        query_content,
                        st.session_state.selected_grade,
                        st.session_state.selected_subject,
                        st.session_state.selected_topic,
                        st.session_state.selected_language,
                        image_b64
                    )
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as error:
                    st.error(f"Error: {error}")
