import cv2
import os
import numpy as np

def crop_dynamic_fundus(image_path, output_size=1400, margin=1):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Can't read image: {image_path}")

    # Convert to grayscale and create a mask for non-black pixels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # Mask everything that isn't black

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError(f"No contours found in the image: {image_path}")

    # Get bounding box around the largest contour (fundus region)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Expand the bounding box a little bit (by the margin)
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, img.shape[1])
    y2 = min(y + h + margin, img.shape[0])

    cropped = img[y1:y2, x1:x2]

    # Resize while maintaining aspect ratio
    h, w = cropped.shape[:2]
    if h > w:
        # Height is greater, so we scale based on height
        new_h = output_size
        new_w = int((w / h) * new_h)
    else:
        # Width is greater, so we scale based on width
        new_w = output_size
        new_h = int((h / w) * new_w)

    resized = cv2.resize(cropped, (new_w, new_h))

    # Create a square black canvas
    top = (output_size - new_h) // 2
    bottom = output_size - new_h - top
    left = (output_size - new_w) // 2
    right = output_size - new_w - left

    # Add padding to make the image square
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded


def process_folder(input_folder, output_folder, output_size=1400, margin=1):
    os.makedirs(output_folder, exist_ok=True)
    supported_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    for filename in os.listdir(input_folder):
        if any(filename.lower().endswith(ext) for ext in supported_exts):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                cropped_img = crop_dynamic_fundus(input_path, output_size, margin)
                cv2.imwrite(output_path, cropped_img)
                print(f"✅ Saved: {output_path}")
            except Exception as e:
                print(f"❌ Skipped {filename}: {e}")


# Run the function
input_folder = "../data/Test_Set/Test"
output_folder = "../data/Test_Set/test_cropped"
process_folder(input_folder, output_folder, output_size=1400, margin=1)
