"""
Utility functions for the IMDb Sentiment Analyzer
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text input for better model performance
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Cleaned and preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove excessive punctuation (more than 3 consecutive)
    text = re.sub(r'[.!?]{4,}', '...', text)
    text = re.sub(r'[,;:]{2,}', ',', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
    
    # Remove excessive capitalization (more than 3 consecutive caps)
    text = re.sub(r'[A-Z]{4,}', lambda m: m.group().capitalize(), text)
    
    return text

def validate_text_input(text: str, max_length: int = 10000) -> Tuple[bool, str]:
    """
    Validate text input for analysis
    
    Args:
        text (str): Input text to validate
        max_length (int): Maximum allowed text length
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Text cannot be empty"
    
    if len(text) > max_length:
        return False, f"Text exceeds maximum length of {max_length} characters"
    
    if len(text.split()) < 3:
        return False, "Text should contain at least 3 words for accurate analysis"
    
    return True, ""

def get_sentiment_details(prediction_proba: np.ndarray, thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Get detailed sentiment analysis results with confidence levels
    
    Args:
        prediction_proba (np.ndarray): Prediction probabilities from model
        thresholds (Dict[str, float]): Confidence thresholds for categorization
        
    Returns:
        Dict[str, Any]: Detailed analysis results
    """
    if thresholds is None:
        thresholds = {
            "very_strong": 0.8,
            "strong": 0.65,
            "moderate": 0.55,
            "weak": 0.0
        }
    
    confidence = max(prediction_proba[0])
    prediction = 1 if prediction_proba[0][1] > 0.5 else 0
    
    # Determine confidence strength
    if confidence >= thresholds["very_strong"]:
        strength = "Very Strong"
        strength_color = "#2E8B57" if prediction == 1 else "#DC143C"
    elif confidence >= thresholds["strong"]:
        strength = "Strong"
        strength_color = "#32CD32" if prediction == 1 else "#FF6347"
    elif confidence >= thresholds["moderate"]:
        strength = "Moderate"
        strength_color = "#9ACD32" if prediction == 1 else "#FF7F50"
    else:
        strength = "Weak"
        strength_color = "#FFD700"
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'positive_prob': prediction_proba[0][1],
        'negative_prob': prediction_proba[0][0],
        'strength': strength,
        'strength_color': strength_color,
        'sentiment_label': 'Positive' if prediction == 1 else 'Negative'
    }

def analyze_text_statistics(text: str) -> Dict[str, Any]:
    """
    Analyze basic text statistics
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict[str, Any]: Text statistics
    """
    if not text:
        return {
            'characters': 0,
            'words': 0,
            'sentences': 0,
            'paragraphs': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0
        }
    
    words = text.split()
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    return {
        'characters': len(text),
        'words': len(words),
        'sentences': len(sentences),
        'paragraphs': len(paragraphs),
        'avg_word_length': round(sum(len(word) for word in words) / len(words), 1) if words else 0,
        'avg_sentence_length': round(len(words) / len(sentences), 1) if sentences else 0
    }

def export_analysis_history(history: List[Dict], format: str = 'csv') -> str:
    """
    Export analysis history to specified format
    
    Args:
        history (List[Dict]): Analysis history data
        format (str): Export format ('csv', 'json')
        
    Returns:
        str: Exported data as string
    """
    if not history:
        return ""
    
    df = pd.DataFrame(history)
    
    # Format timestamp for better readability
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Format probabilities as percentages
    if 'confidence' in df.columns:
        df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")
    if 'positive_prob' in df.columns:
        df['positive_prob'] = df['positive_prob'].apply(lambda x: f"{x:.1%}")
    if 'negative_prob' in df.columns:
        df['negative_prob'] = df['negative_prob'].apply(lambda x: f"{x:.1%}")
    
    # Add sentiment labels
    if 'prediction' in df.columns:
        df['sentiment'] = df['prediction'].map({1: 'Positive', 0: 'Negative'})
    
    if format.lower() == 'csv':
        return df.to_csv(index=False)
    elif format.lower() == 'json':
        return df.to_json(orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported export format: {format}")

def calculate_batch_statistics(results: List[Dict]) -> Dict[str, Any]:
    """
    Calculate statistics for batch analysis results
    
    Args:
        results (List[Dict]): Batch analysis results
        
    Returns:
        Dict[str, Any]: Batch statistics
    """
    if not results:
        return {}
    
    total_count = len(results)
    positive_count = sum(1 for r in results if r.get('prediction') == 1)
    negative_count = total_count - positive_count
    
    confidences = [r.get('confidence', 0) for r in results]
    avg_confidence = np.mean(confidences) if confidences else 0
    
    return {
        'total_reviews': total_count,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'positive_percentage': (positive_count / total_count) * 100 if total_count > 0 else 0,
        'negative_percentage': (negative_count / total_count) * 100 if total_count > 0 else 0,
        'average_confidence': avg_confidence,
        'min_confidence': min(confidences) if confidences else 0,
        'max_confidence': max(confidences) if confidences else 0
    }

def format_confidence_display(confidence: float) -> str:
    """
    Format confidence score for display
    
    Args:
        confidence (float): Confidence score (0-1)
        
    Returns:
        str: Formatted confidence string
    """
    percentage = confidence * 100
    
    if percentage >= 90:
        return f"{percentage:.1f}% (Excellent)"
    elif percentage >= 80:
        return f"{percentage:.1f}% (Very Good)"
    elif percentage >= 70:
        return f"{percentage:.1f}% (Good)"
    elif percentage >= 60:
        return f"{percentage:.1f}% (Fair)"
    else:
        return f"{percentage:.1f}% (Poor)"

def log_analysis(text_length: int, prediction: int, confidence: float):
    """
    Log analysis for monitoring and debugging
    
    Args:
        text_length (int): Length of analyzed text
        prediction (int): Model prediction (0 or 1)
        confidence (float): Prediction confidence
    """
    logger.info(
        f"Analysis completed - Text length: {text_length}, "
        f"Prediction: {'Positive' if prediction == 1 else 'Negative'}, "
        f"Confidence: {confidence:.3f}"
    )