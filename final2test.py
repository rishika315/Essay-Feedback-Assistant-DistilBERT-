# essay_coach_pdf.py
"""
Explainable Essay Feedback Assistant (DistilBERT fine-tuned model + PDF upload)
"""

import streamlit as st
import nltk
import re
import string
import pickle
import time
import textstat
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import language_tool_python
from nltk.tokenize import PunktSentenceTokenizer
from PyPDF2 import PdfReader
import torch

# -----------------------------
# Setup
# -----------------------------
nltk.download('punkt', quiet=True)
punkt_tokenizer = PunktSentenceTokenizer()
lang_tool = language_tool_python.LanguageTool('en-US')

# Load fine-tuned DistilBERT model (your 'model1')
MODEL_PATH = "C:\\Users\\91986\\OneDrive\\Documents\\PROJECTS\\PROJECTS\\nndl\\asap_transformer_model2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# -----------------------------
# Helper functions
# -----------------------------
VAGUE_PHRASES = [
    "very good", "many way", "a lot", "things are", "stuff", "in today’s world", "is good",
    "people say", "some people", "many people", "things", "good thing", "bad thing"
]
VAGUE_PATTERN = re.compile(r'\b(' + r'|'.join([re.escape(p) for p in VAGUE_PHRASES]) + r')\b', re.IGNORECASE)

def extract_text_from_pdf(uploaded_pdf):
    reader = PdfReader(uploaded_pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def preprocess_text(text):
    sentences = punkt_tokenizer.tokenize(text)
    words = []
    for sent in sentences:
        sent_words = [w.strip(string.punctuation) for w in sent.split() if w.strip(string.punctuation)]
        words.extend(sent_words)
    return sentences, words

def grammar_check(text):
    matches = lang_tool.check(text)
    issues = []
    for m in matches:
        issues.append({
            'message': m.message,
            'context': text[max(0, m.offset-30): m.offset+m.errorLength+30],
            'replacements': m.replacements[:3]
        })
    return issues

def detect_vague_phrases(text):
    return [(m.group(0), m.start(), m.end()) for m in VAGUE_PATTERN.finditer(text)]

def readability_metrics(text):
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "smog_index": textstat.smog_index(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text)
    }

def lexical_richness(words):
    if len(words)==0: return 0.0
    types = len(set([w.lower() for w in words if w.isalpha()]))
    return types / len(words)

def score_essay_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.squeeze().item()
    # Map back to ~1–6 scale for interpretability
    scaled_score = round(1 + 5 * max(0, min(score, 1)), 2)
    return scaled_score

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Essay Feedback Assistant (PDF Support)", layout="wide")
st.title("📝 Explainable Essay Feedback Assistant (DistilBERT + PDF Upload)")
st.caption("Upload your essay as text or PDF for instant scoring and analysis.")

option = st.radio("Choose input type:", ["✏️ Type/Paste Essay", "📄 Upload PDF"])

if option == "✏️ Type/Paste Essay":
    input_text = st.text_area("Paste your essay here", height=200)
else:
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    input_text = extract_text_from_pdf(uploaded_pdf) if uploaded_pdf else ""

analyze_btn = st.button("Analyze Essay")

if analyze_btn:
    if len(input_text.strip()) < 20:
        st.warning("Please enter or upload a longer essay (20+ chars).")
        st.stop()

    t0 = time.time()
    sentences, words = preprocess_text(input_text)
    grammar_issues = grammar_check(input_text)
    vague_phrases = detect_vague_phrases(input_text)
    readability = readability_metrics(input_text)
    ttr = lexical_richness(words)
    score = score_essay_bert(input_text)

    # -----------------------------
    # Micro-tips & fun feedback
    # -----------------------------
    tips = []
    if len(grammar_issues) > len(words)/20:
        tips.append("Grammar check: punctuation, spelling, subject-verb agreement.")
    if ttr < 0.05:
        tips.append("Increase vocabulary diversity.")
    if len(vague_phrases) > 0:
        tips.append("Avoid vague phrases; use specific examples.")
    if readability['flesch_reading_ease'] < 40:
        tips.append("Simplify your sentences for clarity.")
    if len(tips) == 0:
        tips.append("Great structure! Consider adding examples for higher clarity.")

    avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
    if avg_sentence_len > 25:
        tips.append("Some sentences are long noodles 🍜 — try splitting them.")
    elif avg_sentence_len < 8:
        tips.append("Sentences are too short ⚡ — try combining them smoothly.")

    paragraphs = [p for p in input_text.split("\n") if p.strip()]
    if len(paragraphs) < 2:
        tips.append("Everything’s in one big chunk 🧱 — split into paragraphs.")

    for cliché in ["at the end of the day", "the world we live in", "every coin has two sides"]:
        if cliché in input_text.lower():
            tips.append(f"Classic cliché alert 🚨 — rephrase '{cliché}' for freshness.")

    # -----------------------------
    # Output section
    # -----------------------------
    st.subheader("📊 DistilBERT Essay Score")
    st.metric("Predicted Score", score)

    st.subheader("📖 Readability Metrics")
    st.json(readability)

    st.subheader("🔤 Lexical Richness (TTR)")
    st.write(round(ttr, 3))

    if grammar_issues:
        st.subheader("🧩 Grammar Issues (top 5)")
        for g in grammar_issues[:5]:
            rep = ", ".join(g['replacements']) if g['replacements'] else "no suggestion"
            st.markdown(f"- {g['message']} — context: _{g['context']}_ — suggestions: *{rep}*")

    if vague_phrases:
        st.subheader("💭 Vague Phrases Detected")
        for p, s, e in vague_phrases:
            st.write(f"- `{p}` at {s}-{e}")

    st.subheader("💡 Actionable Micro-Tips")
    for t in tips:
        st.write("•", t)

    t1 = time.time()
    st.caption(f"Analysis time: {t1 - t0:.1f}s")
