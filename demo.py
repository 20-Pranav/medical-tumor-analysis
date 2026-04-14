import subprocess
import time
import os

print("=" * 60)
print("🧠 Medical Tumor Analysis - Complete Demonstration")
print("=" * 60)

# Step 1: Generate test data
print("\n[1/4] Generating synthetic test data...")
subprocess.run(["python", "make_test_data.py"])

# Step 2: Test model predictions
print("\n[2/4] Testing model on generated volumes...")
subprocess.run(["python", "check_prediction.py"])

# Step 3: Launch Streamlit app
print("\n[3/4] Launching web application...")
print("📌 Upload these files in the web interface:")
print("   - OLD MRI: old_large.nii.gz")
print("   - NEW MRI: stable_new.nii.gz")
print("\n⏳ Starting Streamlit in 3 seconds...")
time.sleep(3)

# Launch the app
subprocess.run(["streamlit", "run", "app.py"])
