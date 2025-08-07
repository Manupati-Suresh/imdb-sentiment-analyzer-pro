"""
Configuration settings for the IMDb Sentiment Analyzer
"""

import os
from pathlib import Path

# Application settings
APP_TITLE = "IMDb Sentiment Analyzer Pro"
APP_ICON = "🎬"
VERSION = "2.0.0"

# Model settings
MODEL_DIR = Path("model")
MODEL_FILE = MODEL_DIR / "imdb_model.pkl"
VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.pkl"

# UI settings
MAX_TEXT_LENGTH = 10000
DEFAULT_TEXT_HEIGHT = 200
BATCH_SIZE_LIMIT = 1000

# Model parameters (for reference)
MODEL_PARAMS = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "stop_words": "english",
    "random_state": 42,
    "max_iter": 1000
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "very_strong": 0.8,
    "strong": 0.65,
    "moderate": 0.55,
    "weak": 0.0
}

# Sample reviews for testing
SAMPLE_REVIEWS = {
    "Positive Example": """This movie was absolutely fantastic! The acting was superb, the plot was engaging, 
    and the cinematography was breathtaking. I would definitely recommend this to anyone looking for a great film experience. 
    The director's vision was clear and the execution was flawless.""",
    
    "Negative Example": """This was one of the worst movies I've ever seen. The plot made no sense, the acting was terrible, 
    and it felt like a complete waste of time. I want my money back. The dialogue was cringe-worthy and the special effects 
    looked like they were done by amateurs.""",
    
    "Mixed Example": """The movie had some good moments with decent special effects, but the story was confusing and the pacing was off. 
    Some parts were entertaining while others dragged on. The lead actor did well but the supporting cast was weak. 
    Overall, it's an average film that could have been much better."""
}

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"