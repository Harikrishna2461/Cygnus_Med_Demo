"""
MLOps & LLMOps Monitoring Module
Tracks system metrics, performance, and resource utilization
"""

import logging
import psutil
import time
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
from threading import Lock
import os
import subprocess

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and aggregate system metrics"""
    
    def __init__(self, window_size=300):
        """
        Initialize metrics collector
        
        Args:
            window_size: Size of time window in seconds for moving averages
        """
        self.window_size = window_size
        self.lock = Lock()
        
        # Request metrics
        self.request_count = defaultdict(int)
        self.request_latencies = defaultdict(lambda: deque(maxlen=100))
        self.request_errors = defaultdict(int)
        
        # Task metrics
        self.task_latencies = {
            'task1_classification': deque(maxlen=100),
            'task2_reasoning': deque(maxlen=100),
            'faiss_query': deque(maxlen=100),
            'llm_call': deque(maxlen=100),
            'stream_buffer': deque(maxlen=100)
        }
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Processing metrics
        self.total_processed_points = 0
        self.total_stream_batches = 0
        self.error_log = deque(maxlen=50)
        
        # System metrics (sampled)
        self.system_samples = deque(maxlen=60)  # 60 samples
        
        # Start time for uptime calculation
        self.start_time = datetime.now()
        self.last_metrics_reset = datetime.now()
    
    def record_request(self, endpoint, latency, success=True):
        """Record API request metrics"""
        with self.lock:
            self.request_count[endpoint] += 1
            self.request_latencies[endpoint].append(latency)
            
            if not success:
                self.request_errors[endpoint] += 1
    
    def record_task_latency(self, task_name, latency):
        """Record individual task latency"""
        with self.lock:
            if task_name in self.task_latencies:
                self.task_latencies[task_name].append(latency)
    
    def record_cache_hit(self):
        """Record cache hit"""
        with self.lock:
            self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        with self.lock:
            self.cache_misses += 1
    
    def record_processed_points(self, count):
        """Record processed data points"""
        with self.lock:
            self.total_processed_points += count
    
    def record_stream_batch(self, batch_size):
        """Record stream batch processing"""
        with self.lock:
            self.total_stream_batches += 1
            self.total_processed_points += batch_size
    
    def record_error(self, endpoint, error_msg):
        """Record error with timestamp"""
        with self.lock:
            self.error_log.append({
                'timestamp': datetime.now().isoformat(),
                'endpoint': endpoint,
                'error': str(error_msg)[:200]  # Truncate long errors
            })
    
    def sample_system_metrics(self):
        """Sample current system metrics"""
        with self.lock:
            try:
                process = psutil.Process(os.getpid())
                
                sample = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': process.cpu_percent(interval=0.1),
                    'memory_mb': process.memory_info().rss / 1024 / 1024,
                    'memory_percent': process.memory_percent(),
                    'open_files': len(process.open_files()),
                    'num_threads': process.num_threads(),
                    'system_memory_percent': psutil.virtual_memory().percent
                }
                
                self.system_samples.append(sample)
                return sample
            except Exception as e:
                logger.error(f"Error sampling system metrics: {e}")
                return None
    
    def get_average_latency(self, task_name):
        """Calculate average latency for task"""
        with self.lock:
            latencies = list(self.task_latencies.get(task_name, []))
            if not latencies:
                return 0.0
            return sum(latencies) / len(latencies)
    
    def get_request_stats(self, endpoint):
        """Get statistics for an endpoint"""
        with self.lock:
            latencies = list(self.request_latencies[endpoint])
            if not latencies:
                return {
                    'count': 0,
                    'errors': 0,
                    'avg_latency': 0.0,
                    'min_latency': 0.0,
                    'max_latency': 0.0,
                    'error_rate': 0.0
                }
            
            return {
                'count': self.request_count[endpoint],
                'errors': self.request_errors[endpoint],
                'avg_latency': sum(latencies) / len(latencies),
                'min_latency': min(latencies),
                'max_latency': max(latencies),
                'error_rate': self.request_errors[endpoint] / self.request_count[endpoint] \
                    if self.request_count[endpoint] > 0 else 0.0
            }
    
    def get_cache_stats(self):
        """Get cache statistics"""
        with self.lock:
            total = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total * 100) if total > 0 else 0.0
            
            return {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'total_accesses': total,
                'hit_rate_percent': hit_rate
            }
    
    def get_system_stats(self):
        """Get current system statistics"""
        with self.lock:
            if not self.system_samples:
                return None
            
            latest = self.system_samples[-1]
            
            # Calculate averages from all samples
            cpu_values = [s['cpu_percent'] for s in self.system_samples]
            memory_values = [s['memory_mb'] for s in self.system_samples]
            
            return {
                'latest': latest,
                'cpu_percent_avg': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'memory_mb_avg': sum(memory_values) / len(memory_values) if memory_values else 0,
                'memory_mb_peak': max(memory_values) if memory_values else 0
            }
    
    def get_uptime(self):
        """Get application uptime"""
        uptime = datetime.now() - self.start_time
        return {
            'start_time': self.start_time.isoformat(),
            'uptime_seconds': int(uptime.total_seconds()),
            'uptime_str': str(uptime).split('.')[0]
        }
    
    def get_summary(self):
        """Get complete metrics summary"""
        with self.lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'uptime': self.get_uptime(),
                'requests': {
                    endpoint: self.get_request_stats(endpoint)
                    for endpoint in self.request_count.keys()
                },
                'tasks': {
                    'task1_avg_latency': self.get_average_latency('task1_classification'),
                    'task2_avg_latency': self.get_average_latency('task2_reasoning'),
                    'faiss_avg_latency': self.get_average_latency('faiss_query'),
                    'llm_avg_latency': self.get_average_latency('llm_call')
                },
                'cache': self.get_cache_stats(),
                'processing': {
                    'total_data_points': self.total_processed_points,
                    'total_stream_batches': self.total_stream_batches
                },
                'system': self.get_system_stats(),
                'recent_errors': list(self.error_log)[-10:]  # Last 10 errors
            }
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self.lock:
            self.request_count.clear()
            self.request_latencies.clear()
            self.request_errors.clear()
            self.task_latencies = {
                'task1_classification': deque(maxlen=100),
                'task2_reasoning': deque(maxlen=100),
                'faiss_query': deque(maxlen=100),
                'llm_call': deque(maxlen=100),
                'stream_buffer': deque(maxlen=100)
            }
            self.cache_hits = 0
            self.cache_misses = 0
            self.total_processed_points = 0
            self.total_stream_batches = 0
            self.last_metrics_reset = datetime.now()
            logger.info("✓ Metrics reset")


class ResourceMonitor:
    """Monitor GPU, storage, and resource usage"""
    
    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        self.faiss_index_path = None
        self.faiss_metadata_path = None
    
    def set_index_paths(self, index_path, metadata_path):
        """Set paths to FAISS index files"""
        self.faiss_index_path = index_path
        self.faiss_metadata_path = metadata_path
    
    def _check_gpu_availability(self):
        """Check if GPU is available"""
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                   capture_output=True, text=True, timeout=5)
            return result.returncode == 0 and 'GPU' in result.stdout
        except:
            return False
    
    def get_gpu_stats(self):
        """Get GPU statistics using nvidia-smi"""
        if not self.gpu_available:
            return {
                'available': False,
                'message': 'No GPU detected or nvidia-smi not available'
            }
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                 '--format=csv,nounits,noheader'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3:
                    memory_used = float(parts[0].strip())
                    memory_total = float(parts[1].strip())
                    gpu_util = float(parts[2].strip())
                    
                    return {
                        'available': True,
                        'memory_used_mb': memory_used,
                        'memory_total_mb': memory_total,
                        'memory_used_percent': (memory_used / memory_total * 100) \
                            if memory_total > 0 else 0,
                        'gpu_utilization_percent': gpu_util
                    }
        except Exception as e:
            logger.error(f"Error getting GPU stats: {e}")
        
        return {
            'available': False,
            'message': 'Error retrieving GPU stats'
        }
    
    def get_storage_stats(self):
        """Get storage statistics for index files"""
        storage = {
            'index_file_mb': 0,
            'metadata_file_mb': 0,
            'total_index_mb': 0
        }
        
        try:
            if self.faiss_index_path and os.path.exists(self.faiss_index_path):
                index_size = os.path.getsize(self.faiss_index_path) / 1024 / 1024
                storage['index_file_mb'] = round(index_size, 2)
            
            if self.faiss_metadata_path and os.path.exists(self.faiss_metadata_path):
                metadata_size = os.path.getsize(self.faiss_metadata_path) / 1024 / 1024
                storage['metadata_file_mb'] = round(metadata_size, 2)
            
            storage['total_index_mb'] = round(
                storage['index_file_mb'] + storage['metadata_file_mb'], 2
            )
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
        
        return storage
    
    def get_disk_usage(self):
        """Get disk usage statistics"""
        try:
            disk = psutil.disk_usage('/')
            return {
                'total_gb': round(disk.total / 1024 / 1024 / 1024, 2),
                'used_gb': round(disk.used / 1024 / 1024 / 1024, 2),
                'free_gb': round(disk.free / 1024 / 1024 / 1024, 2),
                'percent': disk.percent
            }
        except Exception as e:
            logger.error(f"Error getting disk usage: {e}")
            return None
    
    def get_all_resources(self):
        """Get comprehensive resource statistics"""
        return {
            'gpu': self.get_gpu_stats(),
            'storage': self.get_storage_stats(),
            'disk': self.get_disk_usage(),
            'timestamp': datetime.now().isoformat()
        }


class OllamaMonitor:
    """Monitor Ollama model status and performance"""
    
    def __init__(self, ollama_base_url='http://localhost:11434'):
        self.ollama_base_url = ollama_base_url
    
    def get_model_info(self, model_name):
        """Get information about a specific model"""
        try:
            import requests
            
            response = requests.post(
                f"{self.ollama_base_url}/api/show",
                json={"name": model_name},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'name': model_name,
                    'available': True,
                    'details': {
                        'architecture': data.get('details', {}).get('architecture'),
                        'parameters': data.get('details', {}).get('parameter_size'),
                        'quantization': data.get('details', {}).get('quantization_level')
                    },
                    'model_size_mb': round(data.get('model_size', 0) / 1024 / 1024, 2) \
                        if data.get('model_size') else 0
                }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
        
        return {
            'name': model_name,
            'available': False,
            'error': str(e) if 'e' in locals() else 'Unknown error'
        }
    
    def get_models_list(self):
        """Get list of available models"""
        try:
            import requests
            
            response = requests.get(
                f"{self.ollama_base_url}/api/tags",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'available': True,
                    'models': [m.get('name') for m in data.get('models', [])]
                }
        except Exception as e:
            logger.error(f"Error getting models list: {e}")
        
        return {
            'available': False,
            'error': str(e) if 'e' in locals() else 'Unknown error'
        }


# Global instances
metrics_collector = MetricsCollector()
resource_monitor = ResourceMonitor()
ollama_monitor = OllamaMonitor()


def get_all_metrics():
    """Get all metrics for dashboard"""
    return {
        'metrics': metrics_collector.get_summary(),
        'resources': resource_monitor.get_all_resources(),
        'ollama': ollama_monitor.get_models_list()
    }
