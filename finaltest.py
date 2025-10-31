# essay_coach_bert.py
"""
Explainable Essay Feedback Assistant using Fine-Tuned DistilBERT Regressor
"""

import streamlit as st
import nltk
from nltk.tokenize import PunktSentenceTokenizer
import textstat
import re
import string
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import language_tool_python
from collections import Counter

# -----------------------------
# Setup
# -----------------------------
nltk.download('punkt', quiet=True)
punkt_tokenizer = PunktSentenceTokenizer()

# Load fine-tuned DistilBERT model + tokenizer
MODEL_PATH = "C:\\Users\\91986\\OneDrive\\Documents\\PROJECTS\\PROJECTS\\nndl\\asap_transformer_model"  # same folder as your saved model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# LanguageTool for grammar checking
lang_tool = language_tool_python.LanguageTool('en-US')

# -----------------------------
# Helper functions
# -----------------------------
VAGUE_PHRASES = [
    "very good", "many way", "a lot", "things are", "stuff", "in today’s world",
    "is good", "people say", "some people", "many people", "things", "good thing", "bad thing"
]
VAGUE_PATTERN = re.compile(r'\b(' + r'|'.join(map(re.escape, VAGUE_PHRASES)) + r')\b', re.IGNORECASE)

def preprocess_text(text):
    sentences = punkt_tokenizer.tokenize(text.strip())
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
            'offset': m.offset,
            'length': m.errorLength,
            'replacements': m.replacements[:3],
            'context': text[max(0, m.offset-30): m.offset+m.errorLength+30]
        })
    return issues

def detect_vague_phrases(text):
    return [(m.group(0), m.start(), m.end()) for m in VAGUE_PATTERN.finditer(text)]

def readability_metrics(text):
    return {
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'smog_index': textstat.smog_index(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
    }

def lexical_richness(words):
    if not words:
        return 0.0
    types = len(set(w.lower() for w in words if w.isalpha()))
    return types / len(words)

@torch.no_grad()
def score_essay_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    score_norm = outputs.logits.squeeze().item()
    # Assuming normalized scores (0–1), scale back to typical essay range (0–6)
    return round(score_norm * 6, 2)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Essay Feedback Assistant (DistilBERT)", layout="wide")
st.title("🧠 Explainable Essay Feedback Assistant (DistilBERT)")
st.caption("Fine-tuned BERT essay scorer with grammar, clarity, and writing style feedback")

input_text = st.text_area("Paste your essay here", height=200)
analyze_btn = st.button("Analyze Essay")

if analyze_btn:
    if len(input_text.strip()) < 20:
        st.warning("Please enter a longer essay (20+ chars).")
        st.stop()

    t0 = time.time()
    sentences, words = preprocess_text(input_text)
    grammar_issues = grammar_check(input_text)
    vague_phrases = detect_vague_phrases(input_text)
    readability = readability_metrics(input_text)
    ttr = lexical_richness(words)
    bert_score = score_essay_bert(input_text)

    # -----------------------------
    # Feedback logic
    # -----------------------------
    tips = []

    # Base logic
    if len(grammar_issues) > len(words)/20:
        tips.append("Check grammar: punctuation, spelling, subject-verb agreement.")
    if ttr < 0.05:
        tips.append("Increase vocabulary diversity.")
    if len(vague_phrases) > 0:
        tips.append("Avoid vague phrases; use specific examples.")
    if readability['flesch_reading_ease'] < 40:
        tips.append("Simplify long sentences for better clarity.")
    if len(tips) == 0:
        tips.append("Solid essay! Add more concrete examples to raise your score further.")

    # ✨ Fun feedback
    avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
    if avg_sentence_len > 25:
        tips.append("Some sentences feel like long noodles 🍜 — break them up for readability.")
    elif avg_sentence_len < 8:
        tips.append("Your sentences are tiny sparks ⚡ — combine a few for smoother flow.")

    passive_phrases = re.findall(r'\b(is|was|were|be|been|being)\s+\w+ed\b', input_text)
    if len(passive_phrases) > 3:
        tips.append("It’s giving detective novel vibes 🕵️ — reduce passive voice where possible.")

    common_words = [w.lower() for w in words if len(w) > 3]
    freq = Counter(common_words)
    repeated = [w for w, c in freq.items() if c > len(words) * 0.05]
    if repeated:
        tips.append(f"You seem to love '{', '.join(repeated[:3])}' 😅 — try some synonyms!")

    transitions = {"however", "therefore", "moreover", "furthermore", "although", "because", "since", "meanwhile"}
    if not any(t in input_text.lower() for t in transitions):
        tips.append("Add transition words (e.g., 'however', 'therefore') 🧭 for smoother logic flow.")

    emotive_words = {"amazing", "terrible", "wonderful", "awful", "great", "bad", "good", "very"}
    if sum(w.lower() in emotive_words for w in words) > len(words) * 0.03:
        tips.append("Tone check 🎭 — try more neutral phrasing for analytical writing.")

    paragraphs = [p for p in input_text.split("\n") if p.strip()]
    if len(paragraphs) < 2:
        tips.append("It’s a text brick 🧱 — break it into paragraphs for readability.")

    CLICHES = ["at the end of the day", "the world we live in", "every coin has two sides"]
    for c in CLICHES:
        if c in input_text.lower():
            tips.append(f"Classic cliché alert 🚨 — rephrase '{c}' for freshness.")

    # -----------------------------
    # Display Outputs
    # -----------------------------
    st.subheader("📊 DistilBERT Essay Score")
    st.metric("Predicted Score", bert_score)

    st.subheader("📖 Readability Metrics")
    st.json(readability)

    st.subheader("🔤 Lexical Richness (TTR)")
    st.write(round(ttr, 3))

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

    st.caption(f"Analysis time: {time.time()-t0:.1f}s")
