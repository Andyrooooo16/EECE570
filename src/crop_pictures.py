import cv2
import numpy as np
from matplotlib import pyplot as plt

def crop_fundus_circle(image_path, output_size=512):
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur + threshold to isolate the circle
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found.")

    # Assume largest contour is the fundus
    largest_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)

    # Compute crop box
    x = int(x)
    y = int(y)
    r = int(radius)
    x1, x2 = x - r, x + r
    y1, y2 = y - r, y + r

    # Crop & resize
    cropped = img[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (output_size, output_size))

    return resized

# Example usage
img = crop_fundus_circle("example_fundus_image.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Cropped Fundus Image")
plt.axis("off")
plt.show()
