#!/usr/bin/env python3
"""
Health check script for the Streamlit application.
Used by Docker HEALTHCHECK and load balancers.
"""

import sys
import urllib.request
import urllib.error

HEALTH_URL = "http://localhost:8501/_stcore/health"
TIMEOUT = 5


def check_health() -> bool:
    """
    Check if the Streamlit application is healthy.
    
    Returns:
        True if healthy, False otherwise
    """
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=TIMEOUT) as response:
            return response.status == 200
    except urllib.error.URLError:
        return False
    except Exception:
        return False


if __name__ == "__main__":
    if check_health():
        print("Health check: OK")
        sys.exit(0)
    else:
        print("Health check: FAILED")
        sys.exit(1)
