import argparse
import os
import pickle
import glob
import numpy as np
from google.protobuf.json_format import MessageToDict
from scenarionet.converter.waymo.utils import preprocess_waymo_scenarios
from metadrive.type import MetaDriveType

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def get_metadrive_type(waymo_type):
    # Waymo: 1=Vehicle, 2=Pedestrian, 3=Cyclist
    if waymo_type == 1:
        return MetaDriveType.VEHICLE
    elif waymo_type == 2:
        return MetaDriveType.PEDESTRIAN
    elif waymo_type == 3:
        return MetaDriveType.CYCLIST
    else:
        return MetaDriveType.OTHER

def extract_state_arrays(track_proto):
    states = track_proto.get("states", [])
    if not states:
        return None

    # 1. Get Dimensions
    length = float(track_proto.get("length", 4.5))
    width = float(track_proto.get("width", 2.0))
    height = float(track_proto.get("height", 1.5))

    # 2. Extract Dynamic States
    positions = []
    headings = []
    velocities = []
    sizes = []
    valids = []

    for state in states:
        positions.append([state.get("center_x", 0.0), state.get("center_y", 0.0), state.get("center_z", 0.0)])
        headings.append(state.get("heading", 0.0))
        velocities.append([state.get("velocity_x", 0.0), state.get("velocity_y", 0.0)])
        sizes.append([length, width, height])
        valids.append(1)

    count = len(valids)

    # 3. Construct Dictionary (Broadcasting scalars to arrays)
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

    if not points_raw:
        return None

    pts = []
    for p in points_raw:
        pts.append([p.get("x", 0), p.get("y", 0)])
    
    new_feature[geometry_key] = np.array(pts, dtype=np.float32)
    return new_feature

def process_scenario(raw_data):
    if not isinstance(raw_data, dict):
        try:
            raw_data = MessageToDict(raw_data, preserving_proto_field_name=True, use_integers_for_enums=True)
        except:
            return None

    # 1. Determine SDC ID
    sdc_index = raw_data.get('sdc_track_index', 0)
    tracks_list = raw_data.get('tracks', [])
    timestamps = np.array(raw_data.get('timestamps_seconds', []), dtype=np.float32)
    
    sdc_id = "unknown"
    if tracks_list and sdc_index < len(tracks_list):
        sdc_id = str(tracks_list[sdc_index].get('id'))
    
    # 2. Process Tracks
    new_tracks = {}
    for track in tracks_list:
        t_id = str(track.get('id'))
        t_type_raw = track.get('object_type', 1) 
        t_type = get_metadrive_type(t_type_raw) # Map 1 -> "VEHICLE"
        
        state_dict = extract_state_arrays(track)
        
        if t_id and state_dict:
            new_tracks[t_id] = {
                "type": t_type,
                "state": state_dict,
                "metadata": {
                    "track_length": len(state_dict["valid"]),
                    "type": t_type,
                    "object_id": t_id
                }
            }

    # 3. Validate SDC
    if sdc_id not in new_tracks:
        return None

    # 4. Process Map
    raw_map = raw_data.get('map_features', [])
    new_map = {}
    for feature in raw_map:
        processed_feat = process_map_feature(feature)
        if processed_feat:
            f_id = processed_feat["id"]
            new_map[f_id] = processed_feat

    # 5. Final Object
    final_data = {
        "length": len(timestamps), # <--- FIXED: Added scenario length
        "ts": timestamps,          # <--- FIXED: Added timestamps key expected by some managers
        "metadata": {
            "sdc_id": sdc_id,
            "scenario_id": raw_data.get('scenario_id', "unknown"),
            "dataset": "waymo",
        },
        "tracks": new_tracks,
        "map_features": new_map,
        "dynamic_map_states": {}, 
        "id": raw_data.get('scenario_id'),
        "version": "waymo_v1.2"
    }
    
    return final_data

def convert_and_patch(raw_path, output_path):
    print(f"🚀 Starting FINAL COMPLETE Conversion...")
    os.makedirs(output_path, exist_ok=True)

    files = glob.glob(os.path.join(raw_path, "*.tfrecord*"))
    if not files:
        files = glob.glob(os.path.join(raw_path, "**", "*.tfrecord*"), recursive=True)
    files.sort()
    
    if not files:
        print("❌ No .tfrecord files found.")
        return

    try:
        print("⏳ initializing generator...")
        scenario_generator = preprocess_waymo_scenarios(files, 0)
        
        print("⏳ Streaming and converting...")
        count = 0
        skipped = 0
        
        for i, scenario_proto in enumerate(scenario_generator):
            scenario_dict = process_scenario(scenario_proto)
            
            if scenario_dict is None:
                skipped += 1
                continue

            file_name = f"sd_waymo_v1.2_{i}.pkl"
            save_path = os.path.join(output_path, file_name)
            
            with open(save_path, "wb") as f:
                pickle.dump(scenario_dict, f)
            
            count += 1
            if count % 5 == 0:
                print(f"   ...saved {count} scenarios")

        print(f"🎉 Success! Saved {count} scenarios. Skipped {skipped} broken ones.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    convert_and_patch(args.raw, args.out)