#!/usr/bin/env python3
"""Clear all MLOps metrics from database"""
import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'mlops_metrics.db')

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get counts before deletion
    cursor.execute("SELECT COUNT(*) FROM task_runs")
    runs_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM request_metrics")
    metrics_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM task_results")
    results_count = cursor.fetchone()[0]
    
    # Delete all data
    cursor.execute("DELETE FROM request_metrics")
    cursor.execute("DELETE FROM stream_metrics")
    cursor.execute("DELETE FROM task_results")
    cursor.execute("DELETE FROM task_runs")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database cleared!")
    print(f"   Deleted {runs_count} runs")
    print(f"   Deleted {metrics_count} metrics")
    print(f"   Deleted {results_count} results")
else:
    print("✅ No database file exists yet (will be created on first request)")
