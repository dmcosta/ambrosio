#!/usr/bin/env python3
"""
Health check script for Planner Agent
"""

import sys
import redis
import requests
from kafka import KafkaProducer

def check_redis():
    """Check Redis connection"""
    try:
        r = redis.Redis.from_url('redis://redis:6379')
        r.ping()
        return True
    except:
        return False

def check_kafka():
    """Check Kafka connection"""
    try:
        producer = KafkaProducer(bootstrap_servers=['kafka:9092'])
        producer.close()
        return True
    except:
        return False

def check_metrics_server():
    """Check if metrics server is running"""
    try:
        response = requests.get('http://localhost:8080/metrics', timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main health check"""
    checks = [
        ("Redis", check_redis),
        ("Kafka", check_kafka),
        ("Metrics", check_metrics_server)
    ]
    
    failed = []
    for name, check_func in checks:
        if not check_func():
            failed.append(name)
    
    if failed:
        print(f"Health check failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("Health check passed")
        sys.exit(0)

if __name__ == "__main__":
    main()