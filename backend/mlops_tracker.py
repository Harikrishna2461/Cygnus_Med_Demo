"""
MLOps & LLMOps Metrics Tracking System
Persistent storage and analytics for clinical decision support system
"""

import sqlite3
import json
import logging
import time
import uuid
import psutil
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from contextlib import contextmanager

logger = logging.getLogger(__name__)

METRICS_DB_PATH = os.path.join(os.path.dirname(__file__), 'mlops_metrics.db')


class MLOpsTracker:
    """Track and store MLOps metrics for all tasks"""
    
    def __init__(self, db_path=METRICS_DB_PATH):
        self.db_path = db_path
        self.lock = Lock()
        self.current_run = None
        self.run_start_time = None
        self.initialize_db()
    
    @contextmanager
    def get_db(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def initialize_db(self):
        """Create database tables if they don't exist"""
        with self.lock:
            with self.get_db() as conn:
                cursor = conn.cursor()
                
                # Task runs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS task_runs (
                        run_id TEXT PRIMARY KEY,
                        task_name TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        start_time TIMESTAMP NOT NULL,
                        end_time TIMESTAMP,
                        status TEXT DEFAULT 'in_progress',
                        total_duration_ms REAL,
                        input_description TEXT,
                        num_samples INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Individual request metrics
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS request_metrics (
                        metric_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        task_name TEXT NOT NULL,
                        request_number INTEGER,
                        start_time TIMESTAMP NOT NULL,
                        end_time TIMESTAMP NOT NULL,
                        response_time_ms REAL NOT NULL,
                        
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        total_tokens INTEGER,
                        
                        memory_usage_mb REAL,
                        cpu_percent REAL,
                        memory_available_mb REAL,
                        
                        rag_retrieval_ms REAL,
                        llm_inference_ms REAL,
                        post_processing_ms REAL,
                        
                        model_name TEXT,
                        model_type TEXT,
                        
                        input_size_bytes INTEGER,
                        output_size_bytes INTEGER,
                        
                        cached BOOLEAN DEFAULT 0,
                        error TEXT,
                        
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(run_id) REFERENCES task_runs(run_id)
                    )
                ''')
                
                # Stream-specific metrics
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS stream_metrics (
                        stream_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        task_name TEXT NOT NULL,
                        total_points INTEGER NOT NULL,
                        processed_points INTEGER,
                        buffer_interval_sec REAL,
                        total_stream_duration_ms REAL,
                        average_point_duration_ms REAL,
                        min_point_duration_ms REAL,
                        max_point_duration_ms REAL,
                        
                        total_input_tokens INTEGER,
                        total_output_tokens INTEGER,
                        average_tokens_per_point REAL,
                        
                        total_memory_peak_mb REAL,
                        average_cpu_percent REAL,
                        
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(run_id) REFERENCES task_runs(run_id)
                    )
                ''')
                
                # Task output results
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS task_results (
                        result_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        task_name TEXT NOT NULL,
                        request_number INTEGER,
                        output_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(run_id) REFERENCES task_runs(run_id)
                    )
                ''')
                
                logger.info(f"✓ MLOps database initialized at {self.db_path}")
    
    def start_task_run(self, task_name, task_type, description=None, num_samples=None):
        """
        Start tracking a new task run
        
        Args:
            task_name: Name of the task (e.g., 'Clinical Reasoning', 'Probe Guidance')
            task_type: Type (e.g., 'single', 'stream')
            description: Optional description of the run
            num_samples: Number of samples for stream tasks
        
        Returns:
            run_id: Unique identifier for this run
        """
        with self.lock:
            run_id = str(uuid.uuid4())[:8]
            start_time = datetime.now().isoformat()
            
            with self.get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO task_runs 
                    (run_id, task_name, task_type, start_time, input_description, num_samples)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (run_id, task_name, task_type, start_time, description, num_samples))
            
            self.current_run = run_id
            self.run_start_time = time.time()
            logger.debug(f"Started task run: {run_id} for {task_name}")
            
            return run_id
    
    def end_task_run(self, run_id, status='completed'):
        """Mark task run as completed"""
        with self.lock:
            end_time = datetime.now().isoformat()
            
            if self.run_start_time:
                total_duration_ms = (time.time() - self.run_start_time) * 1000
            else:
                total_duration_ms = 0
            
            with self.get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE task_runs 
                    SET end_time = ?, status = ?, total_duration_ms = ?
                    WHERE run_id = ?
                ''', (end_time, status, total_duration_ms, run_id))
            
            logger.debug(f"Ended task run: {run_id} ({total_duration_ms:.2f}ms)")
    
    def record_request_metric(self, run_id, task_name, request_number, metric_dict):
        """
        Record metrics for a single request
        
        Args:
            run_id: The run ID
            task_name: Name of the task
            request_number: Sequential number of this request
            metric_dict: Dictionary containing all metrics
        """
        with self.lock:
            metric_id = str(uuid.uuid4())[:8]
            
            with self.get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO request_metrics 
                    (metric_id, run_id, task_name, request_number, 
                     start_time, end_time, response_time_ms,
                     input_tokens, output_tokens, total_tokens,
                     memory_usage_mb, cpu_percent, memory_available_mb,
                     rag_retrieval_ms, llm_inference_ms, post_processing_ms,
                     model_name, model_type,
                     input_size_bytes, output_size_bytes,
                     cached, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric_id,
                    run_id,
                    task_name,
                    request_number,
                    metric_dict.get('start_time'),
                    metric_dict.get('end_time'),
                    metric_dict.get('response_time_ms', 0),
                    metric_dict.get('input_tokens'),
                    metric_dict.get('output_tokens'),
                    metric_dict.get('total_tokens'),
                    metric_dict.get('memory_usage_mb'),
                    metric_dict.get('cpu_percent'),
                    metric_dict.get('memory_available_mb'),
                    metric_dict.get('rag_retrieval_ms'),
                    metric_dict.get('llm_inference_ms'),
                    metric_dict.get('post_processing_ms'),
                    metric_dict.get('model_name'),
                    metric_dict.get('model_type'),
                    metric_dict.get('input_size_bytes'),
                    metric_dict.get('output_size_bytes'),
                    metric_dict.get('cached', False),
                    metric_dict.get('error')
                ))
            
            return metric_id
    
    def record_stream_metrics(self, run_id, task_name, stream_metrics_dict):
        """Record aggregate metrics for an entire stream"""
        with self.lock:
            stream_id = str(uuid.uuid4())[:8]
            
            with self.get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO stream_metrics
                    (stream_id, run_id, task_name, total_points, processed_points,
                     buffer_interval_sec, total_stream_duration_ms,
                     average_point_duration_ms, min_point_duration_ms, max_point_duration_ms,
                     total_input_tokens, total_output_tokens, average_tokens_per_point,
                     total_memory_peak_mb, average_cpu_percent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stream_id,
                    run_id,
                    task_name,
                    stream_metrics_dict.get('total_points'),
                    stream_metrics_dict.get('processed_points'),
                    stream_metrics_dict.get('buffer_interval_sec'),
                    stream_metrics_dict.get('total_stream_duration_ms'),
                    stream_metrics_dict.get('average_point_duration_ms'),
                    stream_metrics_dict.get('min_point_duration_ms'),
                    stream_metrics_dict.get('max_point_duration_ms'),
                    stream_metrics_dict.get('total_input_tokens'),
                    stream_metrics_dict.get('total_output_tokens'),
                    stream_metrics_dict.get('average_tokens_per_point'),
                    stream_metrics_dict.get('total_memory_peak_mb'),
                    stream_metrics_dict.get('average_cpu_percent')
                ))
            
            return stream_id
    
    def record_task_result(self, run_id, task_name, request_number, result_json):
        """Store task output results"""
        with self.lock:
            result_id = str(uuid.uuid4())[:8]
            
            with self.get_db() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO task_results
                    (result_id, run_id, task_name, request_number, output_json)
                    VALUES (?, ?, ?, ?, ?)
                ''', (result_id, run_id, task_name, request_number, json.dumps(result_json)))
            
            return result_id
    
    # ===== QUERY METHODS =====
    
    def get_task_runs(self, task_name=None):
        """Fetch all runs for a specific task"""
        with self.get_db() as conn:
            cursor = conn.cursor()
            
            if task_name:
                cursor.execute('''
                    SELECT * FROM task_runs 
                    WHERE task_name = ?
                    ORDER BY start_time DESC
                ''', (task_name,))
            else:
                cursor.execute('SELECT * FROM task_runs ORDER BY start_time DESC')
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_run_details(self, run_id):
        """Get detailed information about a specific run"""
        with self.get_db() as conn:
            cursor = conn.cursor()
            
            # Get run metadata
            cursor.execute('SELECT * FROM task_runs WHERE run_id = ?', (run_id,))
            run = dict(cursor.fetchone())
            
            # Get request metrics
            cursor.execute('''
                SELECT * FROM request_metrics 
                WHERE run_id = ?
                ORDER BY request_number
            ''', (run_id,))
            request_metrics = [dict(row) for row in cursor.fetchall()]
            
            # Get stream metrics if applicable
            cursor.execute('''
                SELECT * FROM stream_metrics 
                WHERE run_id = ?
            ''', (run_id,))
            stream_metrics_row = cursor.fetchone()
            stream_metrics = dict(stream_metrics_row) if stream_metrics_row else None
            
            return {
                'run': run,
                'request_metrics': request_metrics,
                'stream_metrics': stream_metrics,
                'total_requests': len(request_metrics)
            }
    
    def get_run_comparison(self, task_name):
        """Get metrics comparison across multiple runs"""
        with self.get_db() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    tr.run_id,
                    tr.task_type,
                    tr.start_time,
                    tr.total_duration_ms,
                    tr.num_samples,
                    COUNT(DISTINCT rm.metric_id) as total_requests,
                    AVG(rm.response_time_ms) as avg_response_time_ms,
                    MAX(rm.response_time_ms) as max_response_time_ms,
                    MIN(rm.response_time_ms) as min_response_time_ms,
                    AVG(rm.total_tokens) as avg_tokens,
                    SUM(rm.total_tokens) as total_tokens,
                    AVG(rm.memory_usage_mb) as avg_memory_mb,
                    AVG(rm.cpu_percent) as avg_cpu_percent
                FROM task_runs tr
                LEFT JOIN request_metrics rm ON tr.run_id = rm.run_id
                WHERE tr.task_name = ?
                GROUP BY tr.run_id
                ORDER BY tr.start_time DESC
            ''', (task_name,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_performance_trend(self, task_name, limit=20):
        """Get performance trend over time"""
        with self.get_db() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    rm.request_number,
                    rm.response_time_ms,
                    rm.total_tokens,
                    rm.memory_usage_mb,
                    rm.cpu_percent,
                    rm.rag_retrieval_ms,
                    rm.llm_inference_ms,
                    rm.post_processing_ms,
                    rm.start_time,
                    rm.cached
                FROM request_metrics rm
                JOIN task_runs tr ON rm.run_id = tr.run_id
                WHERE tr.task_name = ?
                ORDER BY rm.start_time DESC
                LIMIT ?
            ''', (task_name, limit))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]


# Global tracker instance
mlops_tracker = MLOpsTracker()
