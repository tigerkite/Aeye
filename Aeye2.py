import cv2
import numpy as np
import os
import random
from PIL import Image
import tensorflow as tf

def overlay_image(background, overlay, x, y, w, h, alpha=0.5):
    # This function overlays 'overlay' onto 'background' at the position specified
    # by (x, y) and with size (w, h).
    overlay_resized = cv2.resize(overlay, (w, h))
    if overlay_resized.shape[2] == 4:  # Checking for alpha channel.
        overlay_color = overlay_resized[:, :, :3]
        overlay_mask = overlay_resized[:, :, 3] / 255.0
    else:
        overlay_color = overlay_resized
        overlay_mask = np.full((h, w), alpha)

    background_part = background[y:y+h, x:x+w]
    background[y:y+h, x:x+w] = (overlay_color * overlay_mask.reshape(h, w, 1) + 
                                background_part * (1 - overlay_mask.reshape(h, w, 1)))

    return background

def process_image(input_path, output_path):
    # Load the image
    image = Image.open(input_path)
    image = image.rotate(-90, expand=True)
    image_array = np.array(image)

    # Convert to grayscale for OpenCV processing
    image_array_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Load the Eye Detection Haar Cascade XML
    haar_model = cv2.data.haarcascades + 'haarcascade_eye.xml'
    eye_cascade = cv2.CascadeClassifier(haar_model)

    # Check if the cascade is loaded correctly
    if eye_cascade.empty():
        raise ValueError("The Eye Detection Haar Cascade XML file could not be loaded.")

    # Detect eyes in the image
    eyes = eye_cascade.detectMultiScale(image_array_gray, scaleFactor=1.07, minNeighbors=4, minSize=(120, 120))

    for (x, y, w, h) in eyes:
        random_eye_image_path = os.path.join("result", random.choice(os.listdir("result")))
        random_eye_image = cv2.imread(random_eye_image_path, cv2.IMREAD_UNCHANGED)
        if random_eye_image is None:
            print(f"Failed to load image from {random_eye_image_path}")
            continue
        image_array = overlay_image(image_array, random_eye_image, x, y, w, h)

    # Save the processed image
    cv2.imwrite(output_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    # This section is for testing the function directly.
    # When integrated with Flask, this block won't be executed
    # because the name will not be '__main__' when imported as a module.
    process_image("path/to/input/image.jpg", "path/to/output/processed_image.jpg")
