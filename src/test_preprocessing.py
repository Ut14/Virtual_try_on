from image_preprocessing import PreprocessInput
import numpy as np
from PIL import Image
import os

# Initialize the preprocessing class
preprocessor = PreprocessInput()

# Path to the test image
test_image_path = "src/image.png"  # Replace with your actual image path
output_folder = "input"  # Folder where the final processed image will be saved
os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

# Test background removal
img_no_bg = preprocessor.remove_background(test_image_path)

# Check if output is a NumPy array
if isinstance(img_no_bg, np.ndarray):
    print("Background removal successful. Output is a NumPy array.")
else:
    print("Background removal failed.")
    exit()

# Convert NumPy array back to PIL image for resizing
img_no_bg_pil = Image.fromarray(img_no_bg)

# Test resizing
resized_img = preprocessor.resize(img_no_bg_pil)

# Save the final image to the input folder
output_path = os.path.join(output_folder, "processed_image.png")
resized_img.save(output_path)
print(f"Processed image saved at: {output_path}")
