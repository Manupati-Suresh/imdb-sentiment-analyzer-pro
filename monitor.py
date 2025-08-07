"""
Monitoring script for IMDb Sentiment Analyzer
"""

import time
import requests
import logging
import json
import psutil
import os
from datetime import datetime
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AppMonitor:
    def __init__(self, url='http://localhost:8501', check_interval=30):
        self.url = url
        self.check_interval = check_interval
        self.metrics_file = Path('metrics.json')
        self.alerts_sent = set()
        
    def check_health(self):
        """Check application health"""
        try:
            response = requests.get(f"{self.url}/_stcore/health", timeout=10)
            if response.status_code == 200:
                return True, "Healthy"
            else:
                return False, f"HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, str(e)
    
    def check_model_files(self):
        """Check if model files exist and are accessible"""
        model_files = [
            'model/imdb_model.pkl',
            'model/tfidf_vectorizer.pkl'
        ]
        
        missing_files = []
        for file_path in model_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            return False, f"Missing files: {missing_files}"
        return True, "All model files present"
    
    def get_system_metrics(self):
        """Get system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def test_prediction(self):
        """Test model prediction functionality"""
        try:
            # This would require the app to have an API endpoint
            # For now, we'll just check if the health endpoint responds
            health_ok, health_msg = self.check_health()
            return health_ok, health_msg
        except Exception as e:
            return False, str(e)
    
    def save_metrics(self, metrics):
        """Save metrics to file"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = []
            
            all_metrics.append(metrics)
            
            # Keep only last 1000 entries
            if len(all_metrics) > 1000:
                all_metrics = all_metrics[-1000:]
            
            with open(self.metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def send_alert(self, alert_type, message):
        """Send alert (placeholder for actual alerting system)"""
        alert_key = f"{alert_type}_{message}"
        
        if alert_key not in self.alerts_sent:
            logger.error(f"ALERT [{alert_type}]: {message}")
            self.alerts_sent.add(alert_key)
            
            # Here you could integrate with:
            # - Email notifications
            # - Slack webhooks
            # - PagerDuty
            # - SMS alerts
            
    def clear_alert(self, alert_type, message):
        """Clear alert when issue is resolved"""
        alert_key = f"{alert_type}_{message}"
        if alert_key in self.alerts_sent:
            self.alerts_sent.remove(alert_key)
            logger.info(f"RESOLVED [{alert_type}]: {message}")
    
    def check_thresholds(self, metrics):
        """Check if metrics exceed thresholds and send alerts"""
        
        # CPU threshold
        if metrics.get('cpu_percent', 0) > 80:
            self.send_alert('HIGH_CPU', f"CPU usage: {metrics['cpu_percent']:.1f}%")
        else:
            self.clear_alert('HIGH_CPU', 'CPU usage normal')
        
        # Memory threshold
        if metrics.get('memory_percent', 0) > 85:
            self.send_alert('HIGH_MEMORY', f"Memory usage: {metrics['memory_percent']:.1f}%")
        else:
            self.clear_alert('HIGH_MEMORY', 'Memory usage normal')
        
        # Disk threshold
        if metrics.get('disk_percent', 0) > 90:
            self.send_alert('HIGH_DISK', f"Disk usage: {metrics['disk_percent']:.1f}%")
        else:
            self.clear_alert('HIGH_DISK', 'Disk usage normal')
        
        # Low disk space threshold
        if metrics.get('disk_free_gb', float('inf')) < 1.0:
            self.send_alert('LOW_DISK_SPACE', f"Free disk space: {metrics['disk_free_gb']:.2f} GB")
        else:
            self.clear_alert('LOW_DISK_SPACE', 'Disk space sufficient')
    
    def run_single_check(self):
        """Run a single monitoring check"""
        timestamp = datetime.now().isoformat()
        
        # Check application health
        health_ok, health_msg = self.check_health()
        
        # Check model files
        models_ok, models_msg = self.check_model_files()
        
        # Get system metrics
        system_metrics = self.get_system_metrics()
        
        # Test prediction
        prediction_ok, prediction_msg = self.test_prediction()
        
        # Compile metrics
        metrics = {
            'timestamp': timestamp,
            'health_status': 'healthy' if health_ok else 'unhealthy',
            'health_message': health_msg,
            'models_status': 'ok' if models_ok else 'error',
            'models_message': models_msg,
            'prediction_status': 'ok' if prediction_ok else 'error',
            'prediction_message': prediction_msg,
            **system_metrics
        }
        
        # Check thresholds and send alerts
        self.check_thresholds(metrics)
        
        # Send alerts for critical issues
        if not health_ok:
            self.send_alert('APP_DOWN', health_msg)
        else:
            self.clear_alert('APP_DOWN', 'Application is healthy')
        
        if not models_ok:
            self.send_alert('MODEL_ERROR', models_msg)
        else:
            self.clear_alert('MODEL_ERROR', 'Models are accessible')
        
        # Save metrics
        self.save_metrics(metrics)
        
        # Log status
        status = "✓" if health_ok and models_ok and prediction_ok else "✗"
        logger.info(f"{status} Health: {health_msg}, Models: {models_msg}, "
                   f"CPU: {system_metrics.get('cpu_percent', 'N/A'):.1f}%, "
                   f"Memory: {system_metrics.get('memory_percent', 'N/A'):.1f}%")
        
        return metrics
    
    def run_continuous(self):
        """Run continuous monitoring"""
        logger.info(f"Starting continuous monitoring of {self.url}")
        logger.info(f"Check interval: {self.check_interval} seconds")
        
        try:
            while True:
                self.run_single_check()
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
    
    def generate_report(self):
        """Generate monitoring report"""
        if not self.metrics_file.exists():
            logger.warning("No metrics file found")
            return
        
        with open(self.metrics_file, 'r') as f:
            metrics = json.load(f)
        
        if not metrics:
            logger.warning("No metrics data available")
            return
        
        # Calculate uptime
        total_checks = len(metrics)
        healthy_checks = sum(1 for m in metrics if m['health_status'] == 'healthy')
        uptime_percent = (healthy_checks / total_checks) * 100 if total_checks > 0 else 0
        
        # Calculate average system metrics
        cpu_avg = sum(m.get('cpu_percent', 0) for m in metrics) / total_checks
        memory_avg = sum(m.get('memory_percent', 0) for m in metrics) / total_checks
        
        # Generate report
        report = f"""
=== IMDb Sentiment Analyzer Monitoring Report ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Uptime Statistics:
- Total checks: {total_checks}
- Healthy checks: {healthy_checks}
- Uptime: {uptime_percent:.2f}%

System Performance:
- Average CPU usage: {cpu_avg:.1f}%
- Average Memory usage: {memory_avg:.1f}%

Recent Status:
- Current health: {metrics[-1]['health_status']}
- Last check: {metrics[-1]['timestamp']}
- Models status: {metrics[-1]['models_status']}
"""
        
        print(report)
        
        # Save report to file
        with open('monitoring_report.txt', 'w') as f:
            f.write(report)
        
        logger.info("Report saved to monitoring_report.txt")

def main():
    parser = argparse.ArgumentParser(description='Monitor IMDb Sentiment Analyzer')
    parser.add_argument('--url', default='http://localhost:8501', 
                       help='Application URL to monitor')
    parser.add_argument('--interval', type=int, default=30,
                       help='Check interval in seconds')
    parser.add_argument('--single', action='store_true',
                       help='Run single check instead of continuous monitoring')
    parser.add_argument('--report', action='store_true',
                       help='Generate monitoring report')
    
    args = parser.parse_args()
    
    monitor = AppMonitor(args.url, args.interval)
    
    if args.report:
        monitor.generate_report()
    elif args.single:
        metrics = monitor.run_single_check()
        print(json.dumps(metrics, indent=2))
    else:
        monitor.run_continuous()

if __name__ == "__main__":
    main()