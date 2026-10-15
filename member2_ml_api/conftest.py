"""
conftest.py — Pytest configuration file.
Ensures that 'member2_ml_api/' is always on sys.path
so that 'from app.main import app' works correctly
no matter where pytest is run from.
"""
import sys
import os

# Add the member2_ml_api directory to Python path
sys.path.insert(0, os.path.dirname(__file__))
