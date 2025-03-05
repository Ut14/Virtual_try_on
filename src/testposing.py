import os
import glob
import json
import subprocess
import time

# Define paths
OPENPOSE_PATH = r"D:\ml\tryon\openpose\bin\OpenPoseDemo.exe"
INPUT_FOLDER = r"D:\ml\tryon\input"
OUTPUT_JSON = r"D:\ml\tryon\pose_output"
OUTPUT_IMAGES = r"D:\ml\tryon\pose_images"
MODEL_FOLDER = r"D:\ml\tryon\openpose\models"

# OpenPose body parts mapping
BODY_PARTS = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
    10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
    15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
    20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel"
}

# Ensure output directories exist
os.makedirs(OUTPUT_JSON, exist_ok=True)
os.makedirs(OUTPUT_IMAGES, exist_ok=True)


def run_openpose():
    """
    Runs OpenPose on all images in the input folder and saves pose keypoints in JSON format.
    """
    command = [
        OPENPOSE_PATH,
        "--image_dir", INPUT_FOLDER,
        "--write_json", OUTPUT_JSON,
        "--write_images", OUTPUT_IMAGES,  # ✅ Save visualized pose images
        "--render_pose", "1",  # ✅ Enables pose rendering
        "--display", "0",  # ❌ No need to display the GUI
        "--model_folder", MODEL_FOLDER
    ]

    print("Executing command:", " ".join(command))

    result = subprocess.run(command, capture_output=True, text=True)

    # Print OpenPose output for debugging
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError("OpenPose execution failed! Check error logs above.")

    # Wait for JSON files to be created
    timeout = 10
    while timeout > 0:
        json_files = glob.glob(f"{OUTPUT_JSON}/*.json")
        if json_files:
            print("JSON files found! Proceeding to keypoint extraction...")
            return json_files
        time.sleep(1)
        timeout -= 1

    raise FileNotFoundError("No JSON files found. OpenPose might have failed.")


def extract_keypoints(json_files):
    """
    Extracts body keypoints from the OpenPose JSON output.
    """
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


# Run OpenPose first
json_files = run_openpose()

# Extract keypoints
keypoints = extract_keypoints(json_files)
print("Extracted Keypoints:", keypoints)
