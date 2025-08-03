#!/usr/bin/env python3
"""
Test script to verify Streamlit app functionality
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from prophet import Prophet
        import warnings
        warnings.filterwarnings('ignore')
        
        print("✅ All required modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_generation():
    """Test data generation functionality"""
    try:
        # Add src to path
        sys.path.append('src')
        
        # Test data generation
        from data.generate_synthetic_data import generate_synthetic_data
        
        # Generate a small dataset for testing
        generate_synthetic_data('test_data.csv', years=1)
        
        # Check if file was created
        if os.path.exists('test_data.csv'):
            df = pd.read_csv('test_data.csv')
            print(f"✅ Data generation test passed - {len(df)} records created")
            
            # Clean up test file
            os.remove('test_data.csv')
            return True
        else:
            print("❌ Data generation test failed - file not created")
            return False
            
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_streamlit_app():
    """Test if the Streamlit app can be imported"""
    try:
        # Import the main app
        import streamlit_app
        print("✅ Streamlit app can be imported successfully")
        return True
    except Exception as e:
        print(f"❌ Streamlit app import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Streamlit app tests...\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Data Generation", test_data_generation),
        ("Streamlit App Import", test_streamlit_app)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your app is ready for deployment.")
        print("\nTo run the app locally:")
        print("streamlit run streamlit_app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 