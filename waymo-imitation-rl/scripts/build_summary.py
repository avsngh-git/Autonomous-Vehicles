import os
import glob
import pickle
from tqdm import tqdm

# CONFIG
DATA_DIR = "data/waymo_processed"
SUMMARY_FILE = "dataset_summary.pkl"

def build_summary():
    data_path = os.path.abspath(DATA_DIR)
    print(f"📂 Scanning directory: {data_path}")

    # 1. Find all PKL files
    files = glob.glob(os.path.join(data_path, "*.pkl"))
    # Exclude the summary itself if it exists
    files = [f for f in files if "dataset_summary" not in f]
    files.sort()

    if not files:
        print("❌ No .pkl files found!")
        return

    print(f"✅ Found {len(files)} .pkl files.")
    print("⏳ Building Strict Summary Index...")

    summary = {}
    
    for f_path in tqdm(files):
        try:
            with open(f_path, "rb") as f:
                data = pickle.load(f)
                
            # The ID usually looks like "hash|path/to/tfrecord"
            # This is fine, AS LONG AS we map it to the local .pkl filename
            s_id = data.get("id")
            
            # CRITICAL FIX: We force the 'filename' field to be the local .pkl name
            # This prevents MetaDrive from trying to find the original tfrecord
            local_filename = os.path.basename(f_path)
            
            if s_id:
                summary[s_id] = {
                    "id": s_id,
                    "filename": local_filename, # <--- The Fix
                    "length": data.get("length", 0),
                    "object_summary": {} 
                }
        except Exception as e:
            print(f"⚠️ Error reading {f_path}: {e}")

    # 3. Save the summary file
    out_path = os.path.join(data_path, SUMMARY_FILE)
    with open(out_path, "wb") as f:
        pickle.dump(summary, f)

    print(f"🎉 Summary built successfully!")
    print(f"   Index saved to: {out_path}")
    print(f"   Mapped {len(summary)} scenarios.")

if __name__ == "__main__":
    build_summary()