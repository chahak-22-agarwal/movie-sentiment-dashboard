"""
=============================================================
  Movie Sentiment Analysis — Flask Backend
  Endpoints:
    POST /analyze          — predict sentiment + fetch movie
    GET  /health           — sanity check
=============================================================
"""

import os
import re
import pickle
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── App setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow requests from the HTML frontend

# ─── Config ──────────────────────────────────────────────────────────────────
OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "391b8884")
# 👆  Get a free key at https://www.omdbapi.com/apikey.aspx  (1,000 req/day free)
# Set it via environment variable:   export OMDB_API_KEY=xxxxxxxx
# Or just paste your key directly above (not recommended for production).

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


# ─── Load ML pipeline ────────────────────────────────────────────────────────
PIPELINE_PATH = os.path.join(MODEL_DIR, "sentiment_pipeline.pkl")

print("📦  Loading ML model …", end=" ")
try:
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)
    print("✅  done")
except FileNotFoundError:
    pipeline = None
    print("⚠   Model not found — run model/train.py first")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT PREPROCESSING  (must match what was used during training)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    USE_NLTK = True
except ImportError:
    STOP_WORDS = set()
    USE_NLTK = False


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if USE_NLTK:
        tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
        text = " ".join(tokens)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: fetch movie details from OMDb
# ─────────────────────────────────────────────────────────────────────────────

def fetch_movie(title: str) -> dict | None:
    """
    Call the OMDb API and return a clean movie-detail dict,
    or None if the movie wasn't found.
    """
    if OMDB_API_KEY == "YOUR_OMDB_API_KEY":
        # Return placeholder data when no key is configured
        return {
            "title":  title.title(),
            "year":   "N/A",
            "genre":  "N/A",
            "actors": "N/A",
            "director": "N/A",
            "plot":   "OMDb API key not configured.",
            "poster": "https://via.placeholder.com/300x450?text=No+Poster",
            "imdb_rating": "N/A",
            "imdb_votes":  "N/A",
            "runtime": "N/A",
            "language": "N/A",
        }

    url = "http://www.omdbapi.com/"
    params = {"apikey": OMDB_API_KEY, "t": title, "type": "movie"}

    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"OMDb request error: {e}")
        return None

    if data.get("Response") == "False":
        return None  # movie not found

    poster = data.get("Poster", "")
    if not poster or poster == "N/A":
        poster = "https://via.placeholder.com/300x450?text=No+Poster"

    return {
        "title":       data.get("Title",     title.title()),
        "year":        data.get("Year",       "N/A"),
        "genre":       data.get("Genre",      "N/A"),
        "actors":      data.get("Actors",     "N/A"),
        "director":    data.get("Director",   "N/A"),
        "plot":        data.get("Plot",       "N/A"),
        "poster":      poster,
        "imdb_rating": data.get("imdbRating", "N/A"),
        "imdb_votes":  data.get("imdbVotes",  "N/A"),
        "runtime":     data.get("Runtime",    "N/A"),
        "language":    data.get("Language",   "N/A"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: /health
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": pipeline is not None,
        "omdb_key_set": OMDB_API_KEY != "YOUR_OMDB_API_KEY",
    })


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE: /analyze
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Expects JSON body:
        { "movie": "Inception", "review": "Absolutely mind-blowing film!" }

    Returns JSON:
        {
          "sentiment":   "Positive",
          "confidence":  94.7,
          "label_raw":   1,
          "movie": { title, year, genre, actors, … },
          "reviews_analyzed": [ { text, sentiment, confidence }, … ]
        }
    """
    body = request.get_json(silent=True)

    # ── Validate input ───────────────────────────────────────────────────────
    if not body:
        return jsonify({"error": "Request body must be JSON."}), 400

    movie_name = (body.get("movie") or "").strip()
    review_raw = (body.get("review") or "").strip()

    # Support analyzing multiple reviews at once (array OR single string)
    reviews_input = body.get("reviews")   # optional list
    if reviews_input and isinstance(reviews_input, list):
        review_texts = [str(r).strip()
                        for r in reviews_input if str(r).strip()]
    elif review_raw:
        review_texts = [review_raw]
    else:
        return jsonify({"error": "Please provide a 'review' (string) or 'reviews' (list)."}), 400

    if not movie_name:
        return jsonify({"error": "Please provide a 'movie' name."}), 400

    # ── ML prediction ────────────────────────────────────────────────────────
    if pipeline is None:
        return jsonify({"error": "Model not loaded. Run model/train.py first."}), 503

    reviews_analyzed = []
    pos_count = neg_count = 0

    for text in review_texts:
        clean = preprocess(text)
        proba = pipeline.predict_proba([clean])[0]   # [P(neg), P(pos)]
        label = int(pipeline.predict([clean])[0])     # 0 or 1
        conf = round(float(proba[label]) * 100, 2)
        sent = "Positive" if label == 1 else "Negative"

        reviews_analyzed.append({
            "text":       text,
            "sentiment":  sent,
            "confidence": conf,
            "pos_prob":   round(float(proba[1]) * 100, 2),
            "neg_prob":   round(float(proba[0]) * 100, 2),
        })

        if label == 1:
            pos_count += 1
        else:
            neg_count += 1

    # Overall: majority vote; for single review use that review's data
    total = len(reviews_analyzed)
    if total == 1:
        overall_sentiment = reviews_analyzed[0]["sentiment"]
        overall_confidence = reviews_analyzed[0]["confidence"]
        overall_label = 1 if overall_sentiment == "Positive" else 0
    else:
        overall_label = 1 if pos_count >= neg_count else 0
        overall_sentiment = "Positive" if overall_label == 1 else "Negative"
        overall_confidence = round(
            (pos_count / total * 100) if overall_label == 1
            else (neg_count / total * 100), 2
        )

    # ── Fetch movie details ───────────────────────────────────────────────────
    movie_data = fetch_movie(movie_name)
    if movie_data is None:
        # Graceful degradation: still return sentiment even if movie not found
        movie_data = {
            "title":       movie_name.title(),
            "year":        "N/A",
            "genre":       "N/A",
            "actors":      "N/A",
            "director":    "N/A",
            "plot":        "Movie not found in OMDb database.",
            "poster":      "https://via.placeholder.com/300x450?text=Not+Found",
            "imdb_rating": "N/A",
            "imdb_votes":  "N/A",
            "runtime":     "N/A",
            "language":    "N/A",
        }

    # ── Build response ────────────────────────────────────────────────────────
    return jsonify({
        "sentiment":         overall_sentiment,
        "confidence":        overall_confidence,
        "label_raw":         overall_label,
        "pos_count":         pos_count,
        "neg_count":         neg_count,
        "total_reviews":     total,
        "movie":             movie_data,
        "reviews_analyzed":  reviews_analyzed,
    })


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
