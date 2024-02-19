import boto3
import json
from PIL import Image
import numpy as np
from io import BytesIO
import os
import cv2
import tensorflow as tf
import random

# Initialize boto3 S3 client
s3_client = boto3.client('s3')

# Define your bucket name globally
BUCKET_NAME = 'aeye'

def generate_presigned_url(object_name, expiration=3600):
    try:
        response = s3_client.generate_presigned_url('put_object',
                                                    Params={'Bucket': BUCKET_NAME,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except Exception as e:
        print(e)
        return None
    return response

def fetch_image_from_s3(object_name):
    try:
        s3_response = s3_client.get_object(Bucket=BUCKET_NAME, Key=object_name)
        image_content = s3_response['Body'].read()
        image = Image.open(BytesIO(image_content))
        return image
    except Exception as e:
        print(e)
        return None

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

def get_random_eye_image():
    # Assuming 'result' directory is at the root of your Lambda deployment package
    result_dir = os.path.join(os.getcwd(), "result")
    eye_images = [img for img in os.listdir(result_dir) if img.endswith(".png") or img.endswith(".jpg")]
    if not eye_images:
        raise ValueError("No eye overlay images found in 'result/' directory.")
    random_eye_image_name = random.choice(eye_images)
    return os.path.join(result_dir, random_eye_image_name) 
    
def process_image(object_name):
    image = fetch_image_from_s3(object_name)
    if image is None:
        return "Failed to fetch image for processing."

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
        random_eye_image_path = get_random_eye_image()
        random_eye_image = cv2.imread(random_eye_image_path, cv2.IMREAD_UNCHANGED)
        if random_eye_image is None:
            print(f"Failed to load image from {random_eye_image_path}")
            continue
        image_array = overlay_image(image_array, random_eye_image, x, y, w, h)

    # Save processed image to a BytesIO buffer
    buffer = BytesIO()
    processed_image.save(buffer, format="JPEG")
    buffer.seek(0)

    # Optionally: Save the processed image back to S3
    processed_key = 'processed/' + object_name
    s3_client.upload_fileobj(buffer, BUCKET_NAME, processed_key, ExtraArgs={'ContentType': 'image/jpeg'})
    
    return f"Processed image uploaded to {processed_key}"

def lambda_handler(event, context):
    if 'queryStringParameters' in event and 'fileName' in event['queryStringParameters']:
        file_name = event['queryStringParameters']['fileName']
        url = generate_presigned_url(file_name)
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'url': url})
        }
    elif 'queryStringParameters' in event and 'processImage' in event['queryStringParameters']:
        file_name = event['queryStringParameters']['processImage']
        result = process_image_and_overlay_eyes(file_name)
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'message': result})
        }
    else:
        return {
            'statusCode': 400,
            'body': json.dumps({'message': 'Invalid request'})
        }
