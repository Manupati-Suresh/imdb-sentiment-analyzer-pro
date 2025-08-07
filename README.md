# 🎬 IMDb Sentiment Analyzer Pro

A production-ready sentiment analysis application that uses machine learning to predict whether movie reviews express positive or negative sentiment.

## ✨ Features

- **Real-time Sentiment Analysis**: Instant prediction with confidence scoring
- **Batch Processing**: Analyze multiple reviews at once via CSV upload or manual input
- **Interactive Visualizations**: Confidence charts and sentiment distributions
- **Analysis History**: Track and export your analysis history
- **Text Preprocessing**: Automatic text cleaning and normalization
- **Production Ready**: Docker support, error handling, and monitoring
- **Modern UI**: Beautiful, responsive interface with custom styling

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ 
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd imdb_sentiment_app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if not already done)
   ```bash
   python train_model.py
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d
```

### Using Docker directly

```bash
# Build the image
docker build -t sentiment-analyzer .

# Run the container
docker run -p 8501:8501 -v $(pwd)/model:/app/model sentiment-analyzer
```

## 📊 Model Performance

- **Algorithm**: Logistic Regression with TF-IDF vectorization
- **Training Data**: 25,000 IMDb movie reviews
- **Validation Accuracy**: 87.72%
- **Features**: 10,000 most important words and bigrams
- **Processing**: Automatic text cleaning and preprocessing

## 🎯 Usage

### Single Review Analysis

1. Navigate to the "Single Analysis" tab
2. Enter or select a sample movie review
3. Click "Analyze Sentiment" to get results
4. View confidence scores and detailed metrics

### Batch Analysis

1. Go to the "Batch Analysis" tab
2. Upload a CSV file with a 'review' column, or
3. Enter multiple reviews (one per line)
4. Process and download results

### Analysis History

- View all your previous analyses
- Export history as CSV
- See sentiment distribution charts
- Track confidence score patterns

## 📁 Project Structure

```
imdb_sentiment_app/
├── app.py                 # Main Streamlit application
├── train_model.py         # Model training script
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
├── .streamlit/
│   └── config.toml      # Streamlit configuration
├── model/               # Trained model files
│   ├── imdb_model.pkl
│   └── tfidf_vectorizer.pkl
└── README.md           # This file
```

## 🔧 Configuration

### Environment Variables

- `LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `PYTHONUNBUFFERED`: Enable unbuffered Python output

### Streamlit Configuration

Customize the app appearance by editing `.streamlit/config.toml`:

- Theme colors
- Upload limits
- Server settings
- Browser behavior

## 🛠️ Development

### Adding New Features

1. **Extend the UI**: Modify `app.py` to add new tabs or components
2. **Add Utilities**: Create helper functions in `utils.py`
3. **Update Configuration**: Add new settings to `config.py`
4. **Test Changes**: Run locally before deploying

### Model Improvements

1. **Retrain with More Data**: Expand the training dataset
2. **Try Different Algorithms**: Experiment with other ML models
3. **Feature Engineering**: Add new text features or preprocessing steps
4. **Hyperparameter Tuning**: Optimize model parameters

## 📈 Monitoring and Logging

The application includes comprehensive logging for:

- Model loading and initialization
- Analysis requests and results
- Error tracking and debugging
- Performance monitoring

Logs are written to stdout and can be captured by Docker or deployment platforms.

## 🚀 Production Deployment

### Cloud Platforms

**Streamlit Cloud**
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with automatic updates

**Heroku**
```bash
# Create Procfile
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

**AWS/GCP/Azure**
- Use the provided Dockerfile
- Deploy to container services (ECS, Cloud Run, Container Instances)
- Set up load balancing and auto-scaling as needed

### Performance Optimization

- **Caching**: Model loading is cached using `@st.cache_resource`
- **Batch Processing**: Efficient handling of multiple reviews
- **Memory Management**: Optimized for production workloads
- **Error Handling**: Graceful degradation and user feedback

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- IMDb for providing the movie review dataset
- Scikit-learn for machine learning tools
- Streamlit for the amazing web framework
- Plotly for interactive visualizations

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](../../issues) page
2. Create a new issue with detailed information
3. Include error messages and steps to reproduce

---

**Built with ❤️ using Python, Streamlit, and Machine Learning**