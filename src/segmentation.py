import os
import cv2
import numpy as np
import torch
import json
import time
import subprocess
from networks.u2_net import U2NET  # Ensure U-2-Net is installed and available

# Paths
INPUT_IMAGE = r"D:\ml\tryon\input\user_image.jpg"
MASK_OUTPUT = r"D:\ml\tryon\segmentation_output\mask.png"
REFINED_MASK = r"D:\ml\tryon\segmentation_output\refined_mask.png"
POSE_JSON_OUTPUT = r"D:\ml\tryon\pose_output"
POSE_IMAGE_OUTPUT = r"D:\ml\tryon\pose_images"
OPENPOSE_EXEC = r"D:\ml\tryon\openpose\bin\OpenPoseDemo.exe"
MODEL_FOLDER = r"D:\ml\tryon\openpose\models"

# Ensure directories exist
os.makedirs(POSE_JSON_OUTPUT, exist_ok=True)
os.makedirs(POSE_IMAGE_OUTPUT, exist_ok=True)
os.makedirs(os.path.dirname(MASK_OUTPUT), exist_ok=True)


# ---------------------- STEP 1: Background Removal & Segmentation ----------------------
def segment_human(image_path, output_path):
    """Generates a human mask using U-2-Net."""
    model = U2NET(3, 1)
    model.load_state_dict(torch.load("model/u2net.pth", map_location=torch.device('cpu')))
    model.eval()

    image = cv2.imread(image_path)
    image = cv2.resize(image, (320, 320))  # Resize for U-2-Net

    # Convert image to tensor
    img_tensor = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        mask = model(img_tensor)[0].squeeze().numpy()

    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask)

    print("✅ Background mask generated:", output_path)


# ---------------------- STEP 2: Process Mask for Segmentation ----------------------
def refine_mask(mask_path, output_path):
    """Refines segmentation mask to highlight body parts."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # Find contours (body parts separation)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmented_body = np.zeros_like(mask)
    
    for i, cnt in enumerate(contours):
        cv2.drawContours(segmented_body, [cnt], -1, (i * 30), thickness=cv2.FILLED)

    cv2.imwrite(output_path, segmented_body)
    print("✅ Refined segmentation mask created:", output_path)


# ---------------------- STEP 3: Run Pose Estimation using OpenPose ----------------------
def run_openpose():
    """Runs OpenPose to extract pose keypoints."""
    command = [
        OPENPOSE_EXEC,
        "--image_dir", os.path.dirname(INPUT_IMAGE),
        "--write_json", POSE_JSON_OUTPUT,
        "--write_images", POSE_IMAGE_OUTPUT,
        "--render_pose", "1",
        "--display", "0",
        "--model_folder", MODEL_FOLDER
    ]

    print("🚀 Running OpenPose...")
    result = subprocess.run(command, capture_output=True, text=True)
    
    print("✅ Pose estimation completed.")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    json_files = [f for f in os.listdir(POSE_JSON_OUTPUT) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError("❌ No JSON pose data found. OpenPose might have failed.")

    return os.path.join(POSE_JSON_OUTPUT, json_files[0])


# ---------------------- STEP 4: Extract Pose Keypoints ----------------------
BODY_PARTS = {
    0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
    10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
    15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
    20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel"
}

def extract_keypoints(json_path):
    """Extracts body keypoints from the OpenPose JSON output."""
    with open(json_path, "r") as f:
        data = json.load(f)

    keypoints = {}
    if "people" in data and len(data["people"]) > 0:
        person = data["people"][0]
        body_keypoints = person["pose_keypoints_2d"]

        for i, part in BODY_PARTS.items():
            keypoints[part] = (body_keypoints[i * 3], body_keypoints[i * 3 + 1])

    print("✅ Extracted keypoints:", keypoints)
    return keypoints


# ---------------------- RUN THE PIPELINE ----------------------
if __name__ == "__main__":
    print("\n🔹 Step 1: Background Removal & Segmentation")
    segment_human(INPUT_IMAGE, MASK_OUTPUT)

    print("\n🔹 Step 2: Refining Segmentation Mask")
    refine_mask(MASK_OUTPUT, REFINED_MASK)

    print("\n🔹 Step 3: Running Pose Estimation")
    pose_json_path = run_openpose()

    print("\n🔹 Step 4: Extracting Keypoints")
    keypoints = extract_keypoints(pose_json_path)

    print("\n✅ All steps completed successfully! Ready for clothing warping.")
