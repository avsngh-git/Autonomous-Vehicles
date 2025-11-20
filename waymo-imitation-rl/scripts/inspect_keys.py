import argparse
import os
import glob
from google.protobuf.json_format import MessageToDict
from scenarionet.converter.waymo.utils import preprocess_waymo_scenarios

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def inspect_data(raw_path):
    print(f"🔍 Inspecting Waymo Data Stream...")
    
    files = glob.glob(os.path.join(raw_path, "*.tfrecord*"))
    if not files:
        files = glob.glob(os.path.join(raw_path, "**", "*.tfrecord*"), recursive=True)
    files.sort()

    if not files:
        print("❌ No files found.")
        return

    print("⏳ Initializing generator...")
    scenario_generator = preprocess_waymo_scenarios(files, 0)
    
    print("⏳ pulling first item...")
    
    try:
        # Get the first item
        item = next(scenario_generator)
        
        print(f"\n📦 Type of yielded item: {type(item)}")
        
        # If it's a tuple (common in some versions), print both parts
        if isinstance(item, tuple):
            print(f"❗ Item is a TUPLE of length {len(item)}")
            print(f"   Index 0 type: {type(item[0])}")
            print(f"   Index 1 type: {type(item[1])}")
            # Usually index 1 is the actual data
            obj = item[1]
        else:
            obj = item

        # Convert to Dict to check keys
        try:
            print(f"🔄 Attempting MessageToDict on {type(obj)}...")
            data_dict = MessageToDict(obj, preserving_proto_field_name=True, use_integers_for_enums=True)
            
            print("\n🔑 FOUND KEYS:")
            print("--------------------------------------------------")
            print(list(data_dict.keys()))
            print("--------------------------------------------------")
            
            if "metadata" in data_dict:
                print("✅ 'metadata' key exists!")
            else:
                print("❌ 'metadata' key is MISSING.")
                
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            # Fallback: check dir()
            print(f"   Attributes: {[d for d in dir(obj) if not d.startswith('_')]}")

    except StopIteration:
        print("❌ Generator yielded nothing.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    args = parser.parse_args()
    
    inspect_data(args.raw)