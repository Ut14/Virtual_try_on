import os
import cv2
import json
import numpy as np
import glob

# Define OpenPose directory (use full path)
OPENPOSE_PATH = 'D:/ml/tryon/openpose'

# OpenPose body parts mapping
BODY_PARTS = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
    10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
    15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
    20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel"
}

def run_openpose(input_folder, output_json):
    """
    Runs OpenPose on all images in the input folder and saves pose keypoints in JSON format.
    """
    os.makedirs(output_json, exist_ok=True)  # Ensure output directory exists

    command = (
    f'"{OPENPOSE_PATH}/bin/OpenPoseDemo.exe" '
    f'--image_dir "{input_folder}" '
    f'--write_json "{output_json}" --render_pose 0'
)

    os.system(command)
    print(f"Pose estimation completed. Results saved in {output_json}")

def extract_keypoints(json_folder):
    """
    Extracts body keypoints from the OpenPose JSON output.
    """
    json_files = glob.glob(f"{json_folder}/*.json")
    if len(json_files) == 0:
        raise FileNotFoundError("No JSON files found in pose_output/. OpenPose might have failed.")

    json_file = json_files[0]  # Pick the first JSON file

    with open(json_file, "r") as f:
        data = json.load(f)
    
    keypoints = {}
    if "people" in data and len(data["people"]) > 0:
        person = data["people"][0]  # Get first detected person
        body_keypoints = person["pose_keypoints_2d"]
        
        # Convert list to dictionary format
        for i, part in BODY_PARTS.items():
            keypoints[part] = (body_keypoints[i * 3], body_keypoints[i * 3 + 1])  # x, y coordinates

    return keypoints

# Example Usage
input_folder = "D:/ml/tryon/input"
output_json = "D:/ml/tryon/pose_output"
run_openpose(input_folder, output_json)

# Extract keypoints
keypoints = extract_keypoints(output_json)
print("Extracted Keypoints:", keypoints)
