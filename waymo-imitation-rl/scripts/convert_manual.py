
import argparse
import os
import pickle
import glob
import logging
from scenarionet.converter.waymo.utils import preprocess_waymo_scenarios

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def convert_and_save(raw_path, output_path):
    print(f"🚀 Starting Manual Conversion...")
    os.makedirs(output_path, exist_ok=True)

    # 1. Find Files
    files = glob.glob(os.path.join(raw_path, "*.tfrecord*"))
    if not files:
        files = glob.glob(os.path.join(raw_path, "**", "*.tfrecord*"), recursive=True)
    files.sort()
    
    if not files:
        print("❌ No .tfrecord files found.")
        return
        
    print(f"   Found {len(files)} files to process.")

    # 2. Run Conversion
    try:
        print("⏳ initializing generator...")
        # We pass worker_index=0 to satisfy the signature
        scenario_generator = preprocess_waymo_scenarios(files, 0)
        
        print("⏳ Streaming and saving scenarios...")
        
        count = 0
        for i, scenario in enumerate(scenario_generator):
            # Create a unique filename
            file_name = f"sd_waymo_v1.2_{i}.pkl"
            save_path = os.path.join(output_path, file_name)
            
            with open(save_path, "wb") as f:
                pickle.dump(scenario, f)
            
            count += 1
            if count % 10 == 0:
                print(f"   ...saved {count} scenarios so far")

        if count == 0:
            print("❌ The generator yielded 0 scenarios. Check your .tfrecord file.")
        else:
            print(f"🎉 Success! Saved {count} .pkl files to {output_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    convert_and_save(args.raw, args.out)
