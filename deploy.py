"""
Deployment script for IMDb Sentiment Analyzer
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all required files and dependencies are present"""
    logger.info("Checking deployment requirements...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'model/imdb_model.pkl',
        'model/tfidf_vectorizer.pkl'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        if 'model/imdb_model.pkl' in missing_files:
            logger.error("Please run 'python train_model.py' first to create the model files")
        return False
    
    logger.info("All required files present ✓")
    return True

def install_dependencies():
    """Install Python dependencies"""
    logger.info("Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], check=True, capture_output=True, text=True)
        logger.info("Dependencies installed successfully ✓")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def run_tests():
    """Run test suite"""
    logger.info("Running tests...")
    
    if not Path('test_app.py').exists():
        logger.warning("Test file not found, skipping tests")
        return True
    
    try:
        # Install pytest if not available
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'pytest'
        ], check=True, capture_output=True, text=True)
        
        # Run tests
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'test_app.py', '-v'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("All tests passed ✓")
            return True
        else:
            logger.warning(f"Some tests failed: {result.stdout}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run tests: {e}")
        return False

def build_docker_image(tag='sentiment-analyzer'):
    """Build Docker image"""
    logger.info(f"Building Docker image: {tag}")
    
    try:
        subprocess.run([
            'docker', 'build', '-t', tag, '.'
        ], check=True)
        logger.info(f"Docker image built successfully: {tag} ✓")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build Docker image: {e}")
        return False
    except FileNotFoundError:
        logger.error("Docker not found. Please install Docker to build images.")
        return False

def deploy_local(port=8501):
    """Deploy locally using Streamlit"""
    logger.info(f"Starting local deployment on port {port}...")
    
    try:
        # Start Streamlit app
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', str(port),
            '--server.address', '0.0.0.0'
        ])
    except KeyboardInterrupt:
        logger.info("Local deployment stopped by user")
    except Exception as e:
        logger.error(f"Failed to start local deployment: {e}")

def deploy_docker(port=8501, tag='sentiment-analyzer'):
    """Deploy using Docker"""
    logger.info(f"Starting Docker deployment on port {port}...")
    
    try:
        # Stop existing container if running
        subprocess.run([
            'docker', 'stop', 'sentiment-analyzer-container'
        ], capture_output=True)
        
        subprocess.run([
            'docker', 'rm', 'sentiment-analyzer-container'
        ], capture_output=True)
        
        # Run new container
        subprocess.run([
            'docker', 'run', '-d',
            '--name', 'sentiment-analyzer-container',
            '-p', f'{port}:8501',
            '-v', f'{os.getcwd()}/model:/app/model',
            tag
        ], check=True)
        
        logger.info(f"Docker container started successfully on port {port} ✓")
        logger.info(f"Access the app at: http://localhost:{port}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to deploy with Docker: {e}")
        return False

def deploy_docker_compose():
    """Deploy using Docker Compose"""
    logger.info("Starting Docker Compose deployment...")
    
    try:
        subprocess.run(['docker-compose', 'up', '-d'], check=True)
        logger.info("Docker Compose deployment started successfully ✓")
        logger.info("Access the app at: http://localhost:8501")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to deploy with Docker Compose: {e}")
        return False

def health_check(url='http://localhost:8501', timeout=30):
    """Perform health check on deployed application"""
    logger.info(f"Performing health check on {url}...")
    
    import requests
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/_stcore/health", timeout=5)
            if response.status_code == 200:
                logger.info("Health check passed ✓")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
    
    logger.error("Health check failed - application may not be running properly")
    return False

def create_deployment_info():
    """Create deployment information file"""
    deployment_info = {
        'deployment_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'version': '2.0.0',
        'python_version': sys.version,
        'platform': sys.platform,
        'model_files': {
            'model': 'model/imdb_model.pkl',
            'vectorizer': 'model/tfidf_vectorizer.pkl'
        }
    }
    
    with open('deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    logger.info("Deployment info saved to deployment_info.json")

def main():
    parser = argparse.ArgumentParser(description='Deploy IMDb Sentiment Analyzer')
    parser.add_argument('--mode', choices=['local', 'docker', 'docker-compose'], 
                       default='local', help='Deployment mode')
    parser.add_argument('--port', type=int, default=8501, help='Port to run on')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    parser.add_argument('--skip-deps', action='store_true', help='Skip installing dependencies')
    parser.add_argument('--build-only', action='store_true', help='Only build Docker image')
    parser.add_argument('--health-check', action='store_true', help='Only perform health check')
    
    args = parser.parse_args()
    
    logger.info("Starting deployment process...")
    
    # Health check only
    if args.health_check:
        health_check(f'http://localhost:{args.port}')
        return
    
    # Check requirements
    if not check_requirements():
        logger.error("Deployment requirements not met")
        sys.exit(1)
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            logger.error("Failed to install dependencies")
            sys.exit(1)
    
    # Run tests
    if not args.skip_tests:
        if not run_tests():
            logger.warning("Tests failed, but continuing with deployment")
    
    # Create deployment info
    create_deployment_info()
    
    # Deploy based on mode
    if args.mode == 'local':
        deploy_local(args.port)
        
    elif args.mode == 'docker':
        if not build_docker_image():
            logger.error("Failed to build Docker image")
            sys.exit(1)
        
        if args.build_only:
            logger.info("Docker image built successfully. Use --mode docker to deploy.")
            return
        
        if not deploy_docker(args.port):
            logger.error("Docker deployment failed")
            sys.exit(1)
        
        # Wait a bit for container to start
        time.sleep(5)
        health_check(f'http://localhost:{args.port}')
        
    elif args.mode == 'docker-compose':
        if not deploy_docker_compose():
            logger.error("Docker Compose deployment failed")
            sys.exit(1)
        
        # Wait a bit for services to start
        time.sleep(10)
        health_check('http://localhost:8501')
    
    logger.info("Deployment completed successfully! 🚀")

if __name__ == "__main__":
    main()