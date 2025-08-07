"""
Streamlit Cloud optimized version of IMDb Sentiment Analyzer Pro
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import os
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="IMDb Sentiment Analyzer Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .positive-sentiment {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .negative-sentiment {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load model and vectorizer with error handling for Streamlit Cloud"""
    try:
        # Check if model files exist
        model_path = "model/imdb_model.pkl"
        vectorizer_path = "model/tfidf_vectorizer.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            st.error("⚠️ Model files not found. Please ensure the model files are in the repository.")
            st.info("If you're seeing this on first deployment, the model files might be too large for GitHub. Please check the repository structure.")
            st.stop()
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Models loaded successfully")
        return model, vectorizer
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.info("This might be due to model files being too large for GitHub. Consider using Git LFS or alternative storage.")
        st.stop()

def preprocess_text(text: str) -> str:
    """Clean and preprocess text input"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text

def get_sentiment_details(prediction_proba: np.ndarray) -> Dict[str, Any]:
    """Get detailed sentiment analysis results"""
    confidence = max(prediction_proba[0])
    prediction = 1 if prediction_proba[0][1] > 0.5 else 0
    
    # Determine sentiment category
    if confidence > 0.8:
        strength = "Very Strong"
    elif confidence > 0.65:
        strength = "Strong"
    elif confidence > 0.55:
        strength = "Moderate"
    else:
        strength = "Weak"
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'positive_prob': prediction_proba[0][1],
        'negative_prob': prediction_proba[0][0],
        'strength': strength
    }

def create_confidence_chart(positive_prob: float, negative_prob: float) -> go.Figure:
    """Create a confidence visualization chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Negative', 'Positive'],
            y=[negative_prob * 100, positive_prob * 100],
            marker_color=['#ff4b2b', '#56ab2f'],
            text=[f'{negative_prob:.1%}', f'{positive_prob:.1%}'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Sentiment Confidence Scores",
        yaxis_title="Confidence (%)",
        xaxis_title="Sentiment",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def analyze_text_stats(text: str) -> Dict[str, int]:
    """Analyze basic text statistics"""
    words = text.split()
    sentences = text.split('.')
    
    return {
        'characters': len(text),
        'words': len(words),
        'sentences': len([s for s in sentences if s.strip()]),
        'avg_word_length': round(sum(len(word) for word in words) / len(words), 1) if words else 0
    }

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Try to load models
try:
    model, vectorizer = load_models()
    models_loaded = True
except:
    models_loaded = False

# Header
st.markdown('<h1 class="main-header">🎬 IMDb Sentiment Analyzer Pro</h1>', unsafe_allow_html=True)

if not models_loaded:
    st.error("⚠️ **Model Loading Issue**")
    st.info("""
    The machine learning models couldn't be loaded. This is likely because:
    
    1. **Model files are too large for GitHub** (they're about 50MB each)
    2. **Git LFS is needed** for large file storage
    
    **To fix this:**
    - The models need to be uploaded using Git LFS
    - Or hosted on external storage (AWS S3, Google Drive, etc.)
    - Or retrained directly in the cloud
    
    **For now, you can see the UI and features, but predictions won't work.**
    """)
    
    # Show demo mode
    st.warning("🔧 **Demo Mode Active** - UI features available, but predictions are disabled")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 App Information")
    st.info("""
    This advanced sentiment analyzer uses machine learning to predict whether movie reviews are positive or negative.
    
    **Features:**
    - Real-time sentiment analysis
    - Confidence scoring
    - Text preprocessing
    - Analysis history
    - Batch processing
    - Export capabilities
    """)
    
    st.markdown("### 🎯 Model Performance")
    st.metric("Validation Accuracy", "87.72%")
    st.metric("Training Samples", "25,000")
    st.metric("Model Type", "Logistic Regression")
    
    if models_loaded:
        st.success("✅ Models loaded successfully")
    else:
        st.error("❌ Models not available")
    
    st.markdown("### 📈 Usage Statistics")
    if st.session_state.analysis_history:
        total_analyses = len(st.session_state.analysis_history)
        positive_count = sum(1 for analysis in st.session_state.analysis_history if analysis['prediction'] == 1)
        st.metric("Total Analyses", total_analyses)
        st.metric("Positive Reviews", f"{positive_count} ({positive_count/total_analyses:.1%})")
    else:
        st.metric("Total Analyses", "0")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Analysis", "📝 Batch Analysis", "📊 History", "ℹ️ About"])

with tab1:
    st.markdown("### Enter a Movie Review for Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sample reviews for quick testing
        sample_reviews = {
            "Select a sample...": "",
            "Positive Example": "This movie was absolutely fantastic! The acting was superb, the plot was engaging, and the cinematography was breathtaking. I would definitely recommend this to anyone looking for a great film experience.",
            "Negative Example": "This was one of the worst movies I've ever seen. The plot made no sense, the acting was terrible, and it felt like a complete waste of time. I want my money back.",
            "Mixed Example": "The movie had some good moments with decent special effects, but the story was confusing and the pacing was off. Some parts were entertaining while others dragged on."
        }
        
        selected_sample = st.selectbox("Quick Test with Sample Reviews:", list(sample_reviews.keys()))
        
        if selected_sample != "Select a sample...":
            user_input = st.text_area(
                "Movie Review Text:",
                value=sample_reviews[selected_sample],
                height=200,
                help="Enter or paste a movie review here. The model works best with reviews that are at least a few sentences long."
            )
        else:
            user_input = st.text_area(
                "Movie Review Text:",
                height=200,
                placeholder="Enter your movie review here...",
                help="Enter or paste a movie review here. The model works best with reviews that are at least a few sentences long."
            )
        
        # Text preprocessing options
        with st.expander("🔧 Preprocessing Options"):
            auto_preprocess = st.checkbox("Auto-preprocess text", value=True, help="Automatically clean and preprocess the input text")
            show_processed = st.checkbox("Show processed text", help="Display the text after preprocessing")
    
    with col2:
        if user_input:
            stats = analyze_text_stats(user_input)
            st.markdown("### 📊 Text Statistics")
            st.metric("Characters", stats['characters'])
            st.metric("Words", stats['words'])
            st.metric("Sentences", stats['sentences'])
            st.metric("Avg Word Length", f"{stats['avg_word_length']}")
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🚀 Analyze Sentiment", use_container_width=True)
    
    if analyze_button and user_input:
        if not models_loaded:
            st.error("❌ Cannot analyze - models not loaded. Please check the model files.")
        else:
            with st.spinner("🔄 Analyzing sentiment..."):
                try:
                    # Preprocess text if enabled
                    processed_text = preprocess_text(user_input) if auto_preprocess else user_input
                    
                    if show_processed and auto_preprocess:
                        st.markdown("**Processed Text:**")
                        st.text(processed_text)
                    
                    # Make prediction
                    input_vec = vectorizer.transform([processed_text])
                    prediction_proba = model.predict_proba(input_vec)
                    
                    # Get detailed results
                    results = get_sentiment_details(prediction_proba)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 🎯 Analysis Results")
                    
                    # Main sentiment result
                    if results['prediction'] == 1:
                        st.markdown(f"""
                        <div class="positive-sentiment">
                            <h2>🟢 POSITIVE SENTIMENT</h2>
                            <p>Confidence: {results['confidence']:.1%} ({results['strength']})</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="negative-sentiment">
                            <h2>🔴 NEGATIVE SENTIMENT</h2>
                            <p>Confidence: {results['confidence']:.1%} ({results['strength']})</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Positive Probability", f"{results['positive_prob']:.1%}")
                    with col2:
                        st.metric("Negative Probability", f"{results['negative_prob']:.1%}")
                    with col3:
                        st.metric("Confidence Level", results['strength'])
                    
                    # Confidence chart
                    fig = create_confidence_chart(results['positive_prob'], results['negative_prob'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save to history
                    analysis_record = {
                        'timestamp': datetime.now(),
                        'text': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                        'prediction': results['prediction'],
                        'confidence': results['confidence'],
                        'positive_prob': results['positive_prob'],
                        'negative_prob': results['negative_prob']
                    }
                    st.session_state.analysis_history.append(analysis_record)
                    
                    st.success("✅ Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}")
    
    elif analyze_button and not user_input:
        st.warning("⚠️ Please enter some text to analyze.")

with tab2:
    st.markdown("### 📝 Batch Analysis")
    
    if not models_loaded:
        st.warning("⚠️ Batch analysis requires models to be loaded.")
    else:
        st.info("Upload a CSV file with movie reviews or enter multiple reviews separated by lines.")
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Upload CSV file with reviews",
            type=['csv'],
            help="CSV should have a column named 'review' containing the movie reviews"
        )
        
        # Manual input option
        st.markdown("**Or enter multiple reviews (one per line):**")
        batch_input = st.text_area(
            "Multiple Reviews:",
            height=200,
            placeholder="Enter each review on a new line...\nReview 1\nReview 2\nReview 3"
        )
        
        if st.button("🔄 Process Batch", use_container_width=True):
            reviews_to_process = []
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'review' in df.columns:
                        reviews_to_process = df['review'].dropna().tolist()
                    else:
                        st.error("❌ CSV file must contain a 'review' column")
                except Exception as e:
                    st.error(f"❌ Error reading CSV file: {str(e)}")
            
            elif batch_input:
                reviews_to_process = [review.strip() for review in batch_input.split('\n') if review.strip()]
            
            if reviews_to_process:
                progress_bar = st.progress(0)
                results_data = []
                
                for i, review in enumerate(reviews_to_process):
                    try:
                        processed_text = preprocess_text(review)
                        input_vec = vectorizer.transform([processed_text])
                        prediction_proba = model.predict_proba(input_vec)
                        results = get_sentiment_details(prediction_proba)
                        
                        results_data.append({
                            'Review': review[:100] + "..." if len(review) > 100 else review,
                            'Sentiment': 'Positive' if results['prediction'] == 1 else 'Negative',
                            'Confidence': f"{results['confidence']:.1%}",
                            'Positive_Prob': f"{results['positive_prob']:.1%}",
                            'Negative_Prob': f"{results['negative_prob']:.1%}"
                        })
                        
                        progress_bar.progress((i + 1) / len(reviews_to_process))
                        
                    except Exception as e:
                        st.error(f"❌ Error processing review {i+1}: {str(e)}")
                
                if results_data:
                    results_df = pd.DataFrame(results_data)
                    st.markdown("### 📊 Batch Analysis Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        positive_count = len([r for r in results_data if r['Sentiment'] == 'Positive'])
                        st.metric("Positive Reviews", f"{positive_count} ({positive_count/len(results_data):.1%})")
                    with col2:
                        negative_count = len([r for r in results_data if r['Sentiment'] == 'Negative'])
                        st.metric("Negative Reviews", f"{negative_count} ({negative_count/len(results_data):.1%})")
                    with col3:
                        st.metric("Total Processed", len(results_data))
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("⚠️ Please upload a CSV file or enter reviews to process.")

with tab3:
    st.markdown("### 📊 Analysis History")
    
    if st.session_state.analysis_history:
        # Convert history to DataFrame
        history_df = pd.DataFrame(st.session_state.analysis_history)
        history_df['Sentiment'] = history_df['prediction'].map({1: 'Positive', 0: 'Negative'})
        
        # Display history table
        display_df = history_df[['timestamp', 'text', 'Sentiment', 'confidence']].copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        display_df.columns = ['Timestamp', 'Review Text', 'Sentiment', 'Confidence']
        
        st.dataframe(display_df, use_container_width=True)
        
        # History statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            sentiment_counts = history_df['Sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map={'Positive': '#56ab2f', 'Negative': '#ff4b2b'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Confidence distribution histogram
            fig_hist = px.histogram(
                history_df,
                x='confidence',
                nbins=20,
                title="Confidence Score Distribution",
                color_discrete_sequence=['#667eea']
            )
            fig_hist.update_xaxis(title="Confidence Score")
            fig_hist.update_yaxis(title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.analysis_history = []
            st.rerun()
        
        # Export history
        history_csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Export History as CSV",
            data=history_csv,
            file_name=f"sentiment_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("📝 No analysis history yet. Start analyzing some reviews!")

with tab4:
    st.markdown("### ℹ️ About This Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 What This App Does
        This advanced sentiment analysis application uses machine learning to determine whether movie reviews express positive or negative sentiment.
        
        #### 🧠 How It Works
        - **Data**: Trained on 25,000 IMDb movie reviews
        - **Model**: Logistic Regression with TF-IDF vectorization
        - **Accuracy**: 87.72% on validation data
        - **Features**: 10,000 most important words and bigrams
        
        #### 🚀 Key Features
        - Real-time sentiment prediction
        - Confidence scoring and strength assessment
        - Text preprocessing and cleaning
        - Batch processing capabilities
        - Analysis history tracking
        - Interactive visualizations
        - Export functionality
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Model Details
        - **Algorithm**: Logistic Regression
        - **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Max Features**: 10,000
        - **N-gram Range**: 1-2 (unigrams and bigrams)
        - **Stop Words**: English stop words removed
        - **Min Document Frequency**: 2
        
        #### 🎨 Technical Stack
        - **Frontend**: Streamlit
        - **ML Library**: Scikit-learn
        - **Visualization**: Plotly
        - **Data Processing**: Pandas, NumPy
        - **Model Persistence**: Joblib
        
        #### 💡 Tips for Best Results
        - Use complete sentences and proper grammar
        - Include specific details about the movie
        - Longer reviews generally produce more accurate results
        - The model works best with English text
        """)
    
    st.markdown("---")
    
    # GitHub repository link
    st.markdown("""
    ### 🔗 Links & Resources
    
    - **GitHub Repository**: [imdb-sentiment-analyzer-pro](https://github.com/Manupati-Suresh/imdb-sentiment-analyzer-pro)
    - **Documentation**: Complete setup and usage guides
    - **Docker Support**: Ready for containerized deployment
    - **CI/CD Pipeline**: Automated testing and deployment
    
    **Built with ❤️ using Streamlit and Machine Learning**
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "🎬 IMDb Sentiment Analyzer Pro | Deployed on Streamlit Cloud"
    "</div>",
    unsafe_allow_html=True
)