# train_asap_model.py
import pandas as pd
import numpy as np
import re
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
DATA_PATH = "ASAP2_train_sourcetexts.csv"  
ESSAY_COL = "full_text"
SCORE_COL = "score"
MODEL_PATH = "asap_ridge_model.pkl"
EMBED_PATH = "asap_embedding_model.pkl"

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv(DATA_PATH, sep=',')  
print("Columns in dataset:", df.columns.tolist())

# Keep only relevant columns and drop rows with missing text or score
df = df.dropna(subset=[ESSAY_COL, SCORE_COL])
df = df[[ESSAY_COL, SCORE_COL]]
df.rename(columns={ESSAY_COL: "essay", SCORE_COL: "score"}, inplace=True)

# -------------------------
# Preprocess text
# -------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'[^A-Za-z0-9.,!?;:()\'"\s]', '', text)  
    return text.strip()

print("Cleaning essays...")
df['essay_clean'] = [clean_text(e) for e in tqdm(df['essay'], desc="Cleaning")]

# -------------------------
# Encode essays with SentenceTransformer
# -------------------------
print("Encoding essays with SentenceTransformer...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
essay_embeddings = embedding_model.encode(df['essay_clean'].tolist(), show_progress_bar=True)

# -------------------------
# Train a fast CPU model (Ridge Regression)
# -------------------------
X = essay_embeddings
y = df['score'].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

ridge = Ridge(alpha=1.0)
print("Training Ridge Regression on ASAP embeddings (fast on CPU)...")
ridge.fit(X_train, y_train)

# Evaluate
y_pred = ridge.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Test RMSE: {rmse:.3f}")

# -------------------------
# Save model and embedding
# -------------------------
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(ridge, f)
print(f"Ridge Regression model saved to {MODEL_PATH}")

with open(EMBED_PATH, 'wb') as f:
    pickle.dump(embedding_model, f)
print(f"SentenceTransformer embedding saved to {EMBED_PATH}")

print("Training complete.")

