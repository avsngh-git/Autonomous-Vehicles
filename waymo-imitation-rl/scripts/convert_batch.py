import argparse
import os
import glob
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from google.protobuf.json_format import MessageToDict
from scenarionet.converter.waymo.utils import preprocess_waymo_scenarios
from metadrive.type import MetaDriveType

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def extract_state_arrays(track_proto):
    states = track_proto.get("states", [])
    if not states: return None
    length = float(track_proto.get("length", 4.5))
    width = float(track_proto.get("width", 2.0))
    height = float(track_proto.get("height", 1.5))
    
    positions, headings, velocities, sizes, valids = [], [], [], [], []
    for state in states:
        positions.append([state.get("center_x", 0.0), state.get("center_y", 0.0), state.get("center_z", 0.0)])
        headings.append(state.get("heading", 0.0))
        velocities.append([state.get("velocity_x", 0.0), state.get("velocity_y", 0.0)])
        sizes.append([length, width, height])
        valids.append(1)
        
    count = len(valids)
    return {
        "position": np.array(positions, dtype=np.float32),
        "heading": np.array(headings, dtype=np.float32),
        "velocity": np.array(velocities, dtype=np.float32),
        "size": np.array(sizes, dtype=np.float32),
        "valid": np.array(valids, dtype=np.int8),
        "length": np.full(count, length, dtype=np.float32),
        "width": np.full(count, width, dtype=np.float32),
        "height": np.full(count, height, dtype=np.float32)
    }

def process_map_feature(feature):
    f_id = str(feature.get("id"))
    new_feature = {"id": f_id, "type": MetaDriveType.UNSET}
    points_raw = []
    geometry_key = "polyline" 
    
    if "lane" in feature:
        new_feature["type"] = MetaDriveType.LANE_SURFACE_STREET
        points_raw = feature["lane"].get("polyline", [])
    elif "road_line" in feature:
        new_feature["type"] = MetaDriveType.LINE_UNKNOWN
        points_raw = feature["road_line"].get("polyline", [])
    elif "road_edge" in feature:
        new_feature["type"] = MetaDriveType.BOUNDARY_LINE
        points_raw = feature["road_edge"].get("polyline", [])
    elif "crosswalk" in feature:
        new_feature["type"] = MetaDriveType.CROSSWALK
        points_raw = feature["crosswalk"].get("polygon", [])
        geometry_key = "polygon"
    elif "driveway" in feature:
        new_feature["type"] = MetaDriveType.DRIVEWAY
        points_raw = feature["driveway"].get("polygon", [])
        geometry_key = "polygon"

    if not points_raw: return None
    pts = [[p.get("x", 0), p.get("y", 0)] for p in points_raw]
    new_feature[geometry_key] = np.array(pts, dtype=np.float32)
    return new_feature

def process_single_file(file_path, output_dir):
    try:
        base_name = os.path.basename(file_path)
        scenario_generator = preprocess_waymo_scenarios([file_path], 0)
        
        processed_count = 0
        for i, scenario_proto in enumerate(scenario_generator):
            if not isinstance(scenario_proto, dict):
                try:
                    scenario_proto = MessageToDict(scenario_proto, preserving_proto_field_name=True, use_integers_for_enums=True)
                except: continue

            # --- ID SANITIZATION ---
            # Original ID: "hash|path/to/training.tfrecord-00000"
            # Clean ID: "hash" (We split by | and take the first part)
            raw_id = scenario_proto.get('scenario_id', "unknown")
            clean_id = raw_id.split("|")[0] if "|" in raw_id else raw_id
            
            # If clean_id is still weird or empty, make a synthetic one
            if not clean_id or "tfrecord" in clean_id:
                clean_id = f"{base_name}_{i}"

            sdc_index = scenario_proto.get('sdc_track_index', 0)
            tracks_list = scenario_proto.get('tracks', [])
            timestamps = np.array(scenario_proto.get('timestamps_seconds', []), dtype=np.float32)
            
            sdc_id = "unknown"
            if tracks_list and sdc_index < len(tracks_list):
                sdc_id = str(tracks_list[sdc_index].get('id'))

            new_tracks = {}
            for track in tracks_list:
                t_id = str(track.get('id'))
                t_type_raw = track.get('object_type', 1)
                # Map Types
                if t_type_raw == 1: t_type = MetaDriveType.VEHICLE
                elif t_type_raw == 2: t_type = MetaDriveType.PEDESTRIAN
                elif t_type_raw == 3: t_type = MetaDriveType.CYCLIST
                else: t_type = MetaDriveType.OTHER

                state_dict = extract_state_arrays(track)
                if t_id and state_dict:
                    new_tracks[t_id] = {
                        "type": t_type,
                        "state": state_dict,
                        "metadata": {"track_length": len(state_dict["valid"]), "type": t_type, "object_id": t_id}
                    }

            if sdc_id not in new_tracks: continue

            raw_map = scenario_proto.get('map_features', [])
            new_map = {}
            for feature in raw_map:
                processed_feat = process_map_feature(feature)
                if processed_feat: new_map[processed_feat["id"]] = processed_feat

            final_data = {
                "length": len(timestamps),
                "ts": timestamps,
                "metadata": {
                    "sdc_id": sdc_id, 
                    "scenario_id": clean_id, # Use Sanatized ID
                    "dataset": "waymo"
                },
                "tracks": new_tracks,
                "map_features": new_map,
                "dynamic_map_states": {},
                "id": clean_id, # Use Sanatized ID
                "version": "waymo_v1.2"
            }
            
            # Save
            save_name = f"sd_waymo_{clean_id}.pkl"
            with open(os.path.join(output_dir, save_name), "wb") as f:
                pickle.dump(final_data, f)
            processed_count += 1
            
        return processed_count
    except Exception as e:
        return f"Error in {file_path}: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4) 
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = glob.glob(os.path.join(args.raw, "*.tfrecord*"))
    
    print(f"🚀 Starting Batch Conversion (Sanitizing IDs)...")
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_single_file, f, args.out) for f in files]
        
        total_scenarios = 0
        for future in tqdm(futures, total=len(files)):
            result = future.result()
            if isinstance(result, int):
                total_scenarios += result
            else:
                print(result) 

    print(f"🎉 Total Scenarios Converted: {total_scenarios}")

if __name__ == "__main__":
    main()