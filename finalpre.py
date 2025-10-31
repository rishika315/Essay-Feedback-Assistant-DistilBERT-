# essay_coach_asap.py
"""
Explainable Essay Feedback Assistant with ASAP-trained Ridge model
"""

import streamlit as st
import nltk
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
import textstat
import re
import pickle
import string
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -----------------------------
# Ensure NLTK punkt data
# -----------------------------
nltk.download('punkt', quiet=True)

# -----------------------------
# Use standard punkt tokenizer (avoid punkt_tab error)
# -----------------------------
punkt_tokenizer = PunktSentenceTokenizer()

# -----------------------------
# Load ASAP-trained Ridge model and embedding
# -----------------------------
MODEL_PATH = "asap_ridge_model.pkl"
EMBED_PATH = "asap_embedding_model.pkl"

with open(MODEL_PATH, 'rb') as f:
    asap_model = pickle.load(f)

with open(EMBED_PATH, 'rb') as f:
    embed_model = pickle.load(f)

# -----------------------------
# Language tool (grammar check)
# -----------------------------
import language_tool_python
lang_tool = language_tool_python.LanguageTool('en-US')

# -----------------------------
# Helper functions
# -----------------------------
VAGUE_PHRASES = [
    "very good", "many way", "a lot", "things are", "stuff", "in today’s world", "is good",
    "people say", "some people", "many people", "things", "good thing", "bad thing"
]
VAGUE_PATTERN = re.compile(r'\b(' + r'|'.join([re.escape(p) for p in VAGUE_PHRASES]) + r')\b', re.IGNORECASE)

def preprocess_text(text):
    text = text.strip()
    # Use standard PunktSentenceTokenizer
    sentences = punkt_tokenizer.tokenize(text)
    # Simple word tokenizer (avoids nltk.word_tokenize which calls sent_tokenize)
    words = []
    for sent in sentences:
        # remove punctuation and split
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
    if len(words)==0: return 0.0
    types = len(set([w.lower() for w in words if w.isalpha()]))
    return types / len(words)

def score_essay_asap(text):
    text_clean = text.strip()
    embedding = embed_model.encode([text_clean])
    score = asap_model.predict(embedding)[0]
    return round(float(score), 2)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Essay Feedback Assistant", layout="wide")
st.title("📝 Explainable Essay Feedback Assistant (ASAP Ridge)")
st.caption("Fast essay scoring + grammar/vague detection + micro-feedback")

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
    asap_score = score_essay_asap(input_text)

    # Simple micro tips
    tips = []
    if len(grammar_issues) > len(words)/20:
        tips.append("Check grammar: punctuation, spelling, subject-verb agreement.")
    if ttr < 0.05:
        tips.append("Increase vocabulary diversity.")
    if len(vague_phrases) > 0:
        tips.append("Avoid vague phrases; use specific examples.")
    if readability['flesch_reading_ease'] < 40:
        tips.append("Simplify sentences to improve clarity.")
    if len(tips) == 0:
        tips.append("Good essay! Try adding more concrete examples for higher score.")

    # -------------------------------------------------
    # ✨ EXTRA FEEDBACK FEATURES (fun + interpretive)
    # -------------------------------------------------

    # 1️⃣ Sentence length analysis
    avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
    if avg_sentence_len > 25:
        tips.append("Some sentences feel like long noodles 🍜 — break them up for easier reading.")
    elif avg_sentence_len < 8:
        tips.append("Your sentences are tiny sparks ⚡ — try combining them for smoother flow.")

    # 2️⃣ Passive voice detection
    passive_phrases = re.findall(r'\b(is|was|were|be|been|being)\s+\w+ed\b', input_text)
    if len(passive_phrases) > 3:
        tips.append("Your essay sounds a bit like a mystery novel 🔍 — too many passive sentences. Make them active!")

    # 3️⃣ Repetition or redundancy check
    from collections import Counter
    common_words = [w.lower() for w in words if len(w) > 3]
    freq = Counter(common_words)
    repeated = [w for w, c in freq.items() if c > len(words) * 0.05]
    if repeated:
        rep_display = ", ".join(repeated[:3])
        tips.append(f"You really like the words '{rep_display}' 😅 — mix it up a bit!")

    # 4️⃣ Transition word check
    transitions = {"however", "therefore", "moreover", "furthermore", "although", "because", "since", "meanwhile"}
    if not any(t in input_text.lower() for t in transitions):
        tips.append("Add transition words like 'however' or 'therefore' 🧭 to make your ideas flow better.")

    # 5️⃣ Emotional / persuasive tone check
    emotive_words = {"amazing", "terrible", "wonderful", "awful", "great", "bad", "good", "very"}
    if sum(w.lower() in emotive_words for w in words) > len(words) * 0.03:
        tips.append("Tone check 🎭 — feels a bit emotional. Make your arguments more objective.")

    # 6️⃣ Paragraph structure feedback
    paragraphs = [p for p in input_text.split("\n") if p.strip()]
    if len(paragraphs) < 2:
        tips.append("Everything’s in one big chunk 🧱 — split into paragraphs for better readability.")

    # 7️⃣ Cliché detector (for fun!)
    CLICHES = ["at the end of the day", "the world we live in", "every coin has two sides"]
    for c in CLICHES:
        if c in input_text.lower():
            tips.append(f"Classic cliché alert 🚨 — try rephrasing '{c}' for a fresher tone.")

    # -----------------------------
    # Show outputs
    # -----------------------------
    st.subheader("ASAP Ridge Model Score")
    st.metric("Score", asap_score)

    st.subheader("Readability Metrics")
    st.json(readability)

    st.subheader("Lexical Richness (TTR)")
    st.write(round(ttr,3))

    st.subheader("Grammar Issues (top 5)")
    for g in grammar_issues[:5]:
        rep = ", ".join(g['replacements']) if g['replacements'] else "no suggestion"
        st.markdown(f"- {g['message']} — context: _{g['context']}_ — suggestions: *{rep}*")

    if vague_phrases:
        st.subheader("Vague Phrases Detected")
        for p,s,e in vague_phrases:
            st.write(f"- `{p}` at {s}-{e}")

    st.subheader("Actionable Micro-Tips")
    for t in tips:
        st.write("•", t)

    t1 = time.time()
    st.caption(f"Analysis time: {t1-t0:.1f}s")
