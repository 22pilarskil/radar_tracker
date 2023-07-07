import json
import os
import requests
import time
from urllib.parse import unquote
from PIL import Image
from io import BytesIO
from tools import remove_files
import cv2

"""
python3 train.py --img 640 --epochs 5 --data ../dataset.yaml --weights yolov5s.pt
"""
# Create necessary directories
remove_files('dataset_flightradar24/images')
remove_files('dataset_flightradar24/labels')

key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDJsdWp2dndhdmYyMHpiYWY4N3dldWIzIiwib3JnYW5pemF0aW9uSWQiOiJjbDJsdWp2dmxhdmYxMHpiYWF2czE5amVrIiwiYXBpS2V5SWQiOiJjbGplZDNjbDMwMTZ4MDcyMTV6dzk1d24xIiwic2VjcmV0IjoiZGNlMmZkZGJhMjRmOTBhMWU2NzAwZjNiMzUzYWU1MWYiLCJpYXQiOjE2ODc4NzQ4MjgsImV4cCI6MjMxOTAyNjgyOH0.UPyfn9WYljPElyCD70ud0JC--bM0BPkcPcb2UZFWUg4"

os.makedirs('dataset_flightradar24/images', exist_ok=True)
os.makedirs('dataset_flightradar24/labels', exist_ok=True)

with open('export-result.ndjson', 'r') as f:
    lines = f.readlines()

    frame_counter = 0

    for line in lines:
        data = json.loads(line)

        # Extract image URL
        image_url = data['data_row']['row_data']

        # Get the image data via GET request
        response = requests.get(image_url)
        if response.status_code == 200:
            image_data = response.content
            image = Image.open(BytesIO(image_data))
            image.save(f'dataset_flightradar24/images/frame_{frame_counter}.jpg')

        # Delay to avoid IP ban
        time.sleep(0.1)

        # Parse labels and convert to YOLO format
        labels = data['projects'][list(data['projects'].keys())[0]]['labels'][0]['annotations']['objects']
        yolo_labels = []
        for label in labels:
            # Normalize bounding box values (x_center, y_center, width, height)
            x_center = (label['bounding_box']['left'] + label['bounding_box']['width'] / 2) / data['media_attributes'][
                'width']
            y_center = (label['bounding_box']['top'] + label['bounding_box']['height'] / 2) / data['media_attributes'][
                'height']
            width = label['bounding_box']['width'] / data['media_attributes']['width']
            height = label['bounding_box']['height'] / data['media_attributes']['height']

            # Assume 'plane' class is class 0
            yolo_labels.append(f"0 {x_center} {y_center} {width} {height}")

        # Write YOLO labels to file
        with open(f'dataset_flightradar24/labels/frame_{frame_counter}.txt', 'w') as lbl:
            lbl.write('\n'.join(yolo_labels))

        frame_counter += 1
