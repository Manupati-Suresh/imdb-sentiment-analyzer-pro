"""
Test suite for the IMDb Sentiment Analyzer
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    preprocess_text,
    validate_text_input,
    get_sentiment_details,
    analyze_text_statistics,
    calculate_batch_statistics,
    format_confidence_display
)

class TestTextPreprocessing:
    """Test text preprocessing functions"""
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing"""
        text = "This is a great movie!!! I loved it so much."
        result = preprocess_text(text)
        assert result == "This is a great movie... I loved it so much."
    
    def test_preprocess_text_html_removal(self):
        """Test HTML tag removal"""
        text = "This movie is <b>amazing</b> and <i>fantastic</i>!"
        result = preprocess_text(text)
        assert "<b>" not in result
        assert "<i>" not in result
        assert "amazing" in result
        assert "fantastic" in result
    
    def test_preprocess_text_url_removal(self):
        """Test URL removal"""
        text = "Check out this review at https://example.com/review"
        result = preprocess_text(text)
        assert "https://example.com/review" not in result
        assert "Check out this review at" in result
    
    def test_preprocess_text_empty_input(self):
        """Test preprocessing with empty input"""
        assert preprocess_text("") == ""
        assert preprocess_text(None) == ""
        assert preprocess_text("   ") == ""
    
    def test_preprocess_text_excessive_caps(self):
        """Test handling of excessive capitalization"""
        text = "THIS MOVIE IS AMAZING!!!"
        result = preprocess_text(text)
        assert "This" in result
        assert "AMAZING" not in result

class TestTextValidation:
    """Test text validation functions"""
    
    def test_validate_text_input_valid(self):
        """Test validation with valid input"""
        text = "This is a good movie review with enough words."
        is_valid, message = validate_text_input(text)
        assert is_valid is True
        assert message == ""
    
    def test_validate_text_input_empty(self):
        """Test validation with empty input"""
        is_valid, message = validate_text_input("")
        assert is_valid is False
        assert "empty" in message.lower()
    
    def test_validate_text_input_too_short(self):
        """Test validation with too short input"""
        text = "Good movie"
        is_valid, message = validate_text_input(text)
        assert is_valid is False
        assert "3 words" in message
    
    def test_validate_text_input_too_long(self):
        """Test validation with too long input"""
        text = "word " * 2000  # 2000 words
        is_valid, message = validate_text_input(text, max_length=100)
        assert is_valid is False
        assert "maximum length" in message

class TestSentimentAnalysis:
    """Test sentiment analysis functions"""
    
    def test_get_sentiment_details_positive(self):
        """Test sentiment details for positive prediction"""
        # Mock prediction probabilities: [negative_prob, positive_prob]
        prediction_proba = np.array([[0.2, 0.8]])
        
        result = get_sentiment_details(prediction_proba)
        
        assert result['prediction'] == 1
        assert result['confidence'] == 0.8
        assert result['positive_prob'] == 0.8
        assert result['negative_prob'] == 0.2
        assert result['sentiment_label'] == 'Positive'
        assert result['strength'] == 'Very Strong'
    
    def test_get_sentiment_details_negative(self):
        """Test sentiment details for negative prediction"""
        prediction_proba = np.array([[0.9, 0.1]])
        
        result = get_sentiment_details(prediction_proba)
        
        assert result['prediction'] == 0
        assert result['confidence'] == 0.9
        assert result['positive_prob'] == 0.1
        assert result['negative_prob'] == 0.9
        assert result['sentiment_label'] == 'Negative'
        assert result['strength'] == 'Very Strong'
    
    def test_get_sentiment_details_weak_confidence(self):
        """Test sentiment details with weak confidence"""
        prediction_proba = np.array([[0.45, 0.55]])
        
        result = get_sentiment_details(prediction_proba)
        
        assert result['prediction'] == 1
        assert result['confidence'] == 0.55
        assert result['strength'] == 'Weak'

class TestTextStatistics:
    """Test text statistics functions"""
    
    def test_analyze_text_statistics_basic(self):
        """Test basic text statistics"""
        text = "This is a test. It has two sentences."
        
        stats = analyze_text_statistics(text)
        
        assert stats['characters'] == len(text)
        assert stats['words'] == 8
        assert stats['sentences'] == 2
        assert stats['paragraphs'] == 1
        assert stats['avg_word_length'] > 0
        assert stats['avg_sentence_length'] == 4.0
    
    def test_analyze_text_statistics_empty(self):
        """Test text statistics with empty input"""
        stats = analyze_text_statistics("")
        
        assert stats['characters'] == 0
        assert stats['words'] == 0
        assert stats['sentences'] == 0
        assert stats['paragraphs'] == 0
        assert stats['avg_word_length'] == 0
        assert stats['avg_sentence_length'] == 0
    
    def test_analyze_text_statistics_multiline(self):
        """Test text statistics with multiple paragraphs"""
        text = "First paragraph.\n\nSecond paragraph with more text."
        
        stats = analyze_text_statistics(text)
        
        assert stats['paragraphs'] == 2
        assert stats['sentences'] == 2

class TestBatchStatistics:
    """Test batch processing statistics"""
    
    def test_calculate_batch_statistics_mixed(self):
        """Test batch statistics with mixed results"""
        results = [
            {'prediction': 1, 'confidence': 0.8},
            {'prediction': 0, 'confidence': 0.7},
            {'prediction': 1, 'confidence': 0.9},
            {'prediction': 0, 'confidence': 0.6}
        ]
        
        stats = calculate_batch_statistics(results)
        
        assert stats['total_reviews'] == 4
        assert stats['positive_count'] == 2
        assert stats['negative_count'] == 2
        assert stats['positive_percentage'] == 50.0
        assert stats['negative_percentage'] == 50.0
        assert stats['average_confidence'] == 0.75
        assert stats['min_confidence'] == 0.6
        assert stats['max_confidence'] == 0.9
    
    def test_calculate_batch_statistics_empty(self):
        """Test batch statistics with empty results"""
        stats = calculate_batch_statistics([])
        assert stats == {}

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_format_confidence_display_excellent(self):
        """Test confidence display formatting for excellent confidence"""
        result = format_confidence_display(0.95)
        assert "95.0%" in result
        assert "Excellent" in result
    
    def test_format_confidence_display_poor(self):
        """Test confidence display formatting for poor confidence"""
        result = format_confidence_display(0.45)
        assert "45.0%" in result
        assert "Poor" in result
    
    def test_format_confidence_display_good(self):
        """Test confidence display formatting for good confidence"""
        result = format_confidence_display(0.75)
        assert "75.0%" in result
        assert "Good" in result

# Integration tests would go here if we had access to the actual model
class TestIntegration:
    """Integration tests (require model files)"""
    
    @pytest.mark.skipif(
        not os.path.exists("model/imdb_model.pkl"),
        reason="Model file not found"
    )
    def test_model_loading(self):
        """Test that models can be loaded successfully"""
        import joblib
        
        try:
            model = joblib.load("model/imdb_model.pkl")
            vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
            assert model is not None
            assert vectorizer is not None
        except Exception as e:
            pytest.fail(f"Failed to load models: {e}")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])