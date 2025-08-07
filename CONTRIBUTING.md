# Contributing to IMDb Sentiment Analyzer Pro

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional, for containerized development)

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/yourusername/imdb-sentiment-analyzer-pro.git
   cd imdb-sentiment-analyzer-pro
   ```

2. **Set up development environment:**
   ```bash
   make setup-dev
   ```

3. **Train the model (if not already done):**
   ```bash
   make train
   ```

4. **Run tests:**
   ```bash
   make test
   ```

5. **Start the application:**
   ```bash
   make run
   ```

## 🛠️ Development Workflow

### Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Run tests and linting:**
   ```bash
   make test
   make lint
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add batch processing for multiple reviews
fix: resolve model loading error on Windows
docs: update installation instructions
test: add unit tests for text preprocessing
```

## 📋 Coding Standards

### Python Code Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Maximum line length: 100 characters

### Code Quality

- Write docstrings for all functions and classes
- Add type hints where appropriate
- Include error handling and logging
- Write unit tests for new functionality
- Maintain test coverage above 80%

### File Organization

```
imdb_sentiment_app/
├── app.py              # Main Streamlit application
├── train_model.py      # Model training script
├── utils.py           # Utility functions
├── config.py          # Configuration settings
├── test_app.py        # Test suite
├── deploy.py          # Deployment script
├── monitor.py         # Monitoring script
├── requirements.txt   # Python dependencies
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose setup
├── Makefile          # Development commands
├── .streamlit/       # Streamlit configuration
├── model/            # Trained model files
└── docs/             # Documentation
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
python -m pytest test_app.py -v

# Run with coverage
python -m pytest test_app.py --cov=. --cov-report=html
```

### Writing Tests

- Place tests in `test_app.py` or create new test files with `test_` prefix
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- Mock external dependencies when necessary

Example test structure:
```python
def test_preprocess_text_removes_html_tags():
    """Test that HTML tags are properly removed from text"""
    input_text = "This is <b>bold</b> text"
    expected = "This is bold text"
    result = preprocess_text(input_text)
    assert result == expected
```

## 📚 Documentation

### Code Documentation

- Write clear docstrings for all public functions and classes
- Include parameter types and return types
- Provide usage examples for complex functions

Example:
```python
def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text input for better model performance
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Cleaned and preprocessed text
        
    Example:
        >>> preprocess_text("This is <b>great</b>!")
        "This is great!"
    """
```

### README Updates

- Update README.md when adding new features
- Include screenshots for UI changes
- Update installation instructions if dependencies change

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment information:**
   - Operating system
   - Python version
   - Package versions

2. **Steps to reproduce:**
   - Clear, numbered steps
   - Expected vs actual behavior
   - Error messages or logs

3. **Additional context:**
   - Screenshots if applicable
   - Sample data that causes the issue

## 💡 Feature Requests

For new features, please:

1. **Check existing issues** to avoid duplicates
2. **Describe the problem** the feature would solve
3. **Propose a solution** with implementation details
4. **Consider alternatives** and explain why your approach is best

## 🔍 Code Review Process

### For Contributors

- Ensure all tests pass before submitting PR
- Write clear PR descriptions explaining changes
- Respond to review feedback promptly
- Keep PRs focused and reasonably sized

### For Reviewers

- Be constructive and respectful in feedback
- Focus on code quality, performance, and maintainability
- Test the changes locally when possible
- Approve PRs that meet quality standards

## 🏷️ Release Process

1. **Version Bumping:**
   - Follow [Semantic Versioning](https://semver.org/)
   - Update version in `config.py`

2. **Release Notes:**
   - Document new features and bug fixes
   - Include breaking changes and migration notes

3. **Deployment:**
   - Test deployment process
   - Update Docker images
   - Update documentation

## 📞 Getting Help

- **Issues:** Create a GitHub issue for bugs or feature requests
- **Discussions:** Use GitHub Discussions for questions and ideas
- **Email:** Contact maintainers for security issues

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- GitHub contributors page

Thank you for contributing to IMDb Sentiment Analyzer Pro! 🎬✨