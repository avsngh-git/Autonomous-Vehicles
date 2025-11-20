import argparse
import os
import pickle
import glob
import logging
from google.protobuf.json_format import MessageToDict
from scenarionet.converter.waymo.utils import preprocess_waymo_scenarios

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def make_dict(obj):
    """
    Forcefully converts a Protobuf Scenario object into a Python Dictionary.
    """
    # Strategy 1: It's already a dict
    if isinstance(obj, dict):
        return obj

    # Strategy 2: It has a .to_dict() method (Common in wrappers)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()

    # Strategy 3: It's a Google Protobuf object
    try:
        # preserving_proto_field_name is CRITICAL for MetaDrive to recognize keys
        return MessageToDict(obj, preserving_proto_field_name=True, use_integers_for_enums=True)
    except Exception as e:
        print(f"⚠️ Failed to convert using MessageToDict: {e}")
        
    # Strategy 4: Fallback inspection (Last resort)
    # If it's some other weird object, try to look at its attributes
    try:
        return {k: getattr(obj, k) for k in dir(obj) if not k.startswith('_') and not callable(getattr(obj, k))}
    except Exception:
        return obj # Return as-is and pray

def convert_and_save(raw_path, output_path):
    print(f"🚀 Starting Final Conversion...")
    os.makedirs(output_path, exist_ok=True)

    # 1. Find Files
    files = glob.glob(os.path.join(raw_path, "*.tfrecord*"))
    if not files:
        files = glob.glob(os.path.join(raw_path, "**", "*.tfrecord*"), recursive=True)
    files.sort()
    
    if not files:
        print("❌ No .tfrecord files found.")
        return

    print(f"   Found {len(files)} files.")

    try:
        print("⏳ Initializing generator...")
        scenario_generator = preprocess_waymo_scenarios(files, 0)
        
        print("⏳ Streaming, converting, and saving...")
        
        count = 0
        for i, scenario in enumerate(scenario_generator):
            
            # --- CRITICAL FIX: Convert to Dict ---
            scenario_dict = make_dict(scenario)
            
            # Basic Validation
            if not isinstance(scenario_dict, dict):
                print(f"❌ Error: Could not convert scenario {i} to dict. Type is {type(scenario)}")
                continue

            # Save
            file_name = f"sd_waymo_v1.2_{i}.pkl"
            save_path = os.path.join(output_path, file_name)
            
            with open(save_path, "wb") as f:
                pickle.dump(scenario_dict, f)
            
            count += 1
            if count % 10 == 0:
                print(f"   ...saved {count} scenarios")

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