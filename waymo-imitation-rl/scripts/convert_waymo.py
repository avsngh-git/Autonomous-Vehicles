import argparse
import os
import logging
import glob
import inspect
import sys

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def run_conversion(raw_path, output_path):
    print(f"🚀 Starting Robust Waymo Converter")
    print(f"   Input:  {raw_path}")
    print(f"   Output: {output_path}")

    # 1. Verify Input Data
    if not os.path.exists(raw_path):
        print(f"❌ Error: Input directory '{raw_path}' does not exist.")
        return

    # Find .tfrecord files (handle both .tfrecord and .tfexample extensions)
    files = glob.glob(os.path.join(raw_path, "*.tfrecord*"))
    if not files:
        # Try recursive search
        files = glob.glob(os.path.join(raw_path, "**", "*.tfrecord*"), recursive=True)
    
    files.sort()
    
    if not files:
        print("❌ No .tfrecord files found!")
        print("   Please download a file from https://waymo.com/open/download/ (Motion Dataset)")
        return

    print(f"✅ Found {len(files)} file(s).")

    # 2. Import ScenarioNet safely
    try:
        import tensorflow as tf
        # Check TF availability (Critical for Waymo)
        print(f"   TensorFlow Version: {tf.__version__}")
        
        from scenarionet.converter.waymo.utils import preprocess_waymo_scenarios
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Run: uv sync")
        return

    # 3. INSPECT THE FUNCTION SIGNATURE
    # This is the magic step. We stop guessing and look at the code.
    sig = inspect.signature(preprocess_waymo_scenarios)
    print(f"🔍 Library Function Signature: {sig}")
    
    params = sig.parameters
    kwargs = {}

    # 4. Build Arguments Dynamically
    # We only pass arguments that the function actually accepts
    
    if 'overwrite' in params:
        kwargs['overwrite'] = True
    
    # Handle the worker argument naming confusion
    if 'worker_num' in params:
        kwargs['worker_num'] = 2
    elif 'num_workers' in params:
        kwargs['num_workers'] = 2
    elif 'process_num' in params:
        kwargs['process_num'] = 2
        
    print(f"   Constructed Params: {kwargs}")

    # 5. Run Conversion
    try:
        print("⏳ Converting... (This takes memory, please wait)")
        preprocess_waymo_scenarios(files, output_path, **kwargs)
        print("\n🎉 Conversion Successful!")
        print(f"   Data saved to: {output_path}")
        
        # Verify output
        pkl_files = glob.glob(os.path.join(output_path, "*.pkl"))
        print(f"   Generated {len(pkl_files)} scenario files.")
        
    except Exception as e:
        print(f"\n❌ Conversion Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    run_conversion(args.raw, args.out)