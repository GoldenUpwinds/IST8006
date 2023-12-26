import cv2
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def get_image_size(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    return width, height


def json_to_yolo(json_data):
    # Load JSON data
    with open(json_data, 'r') as file:
        data = json.load(file)

    # Process each entry in the JSON data
    for entry in data:
        image_name = entry["name"]
        bbox = entry["bbox"]

        # Read image dimensions
        image_path = os.path.join(ROOT, "data", "age", "images", "test", image_name)
        image_width, image_height = get_image_size(image_path)

        # Convert bbox to YOLO format
        x_center = (bbox["x1"] + bbox["x2"]) / (2 * image_width)
        y_center = (bbox["y1"] + bbox["y2"]) / (2 * image_height)
        bbox_width = (bbox["x2"] - bbox["x1"]) / image_width
        bbox_height = (bbox["y2"] - bbox["y1"]) / image_height

        # Create YOLO format annotation
        yolo_annotation = f"0 {x_center} {y_center} {bbox_width} {bbox_height}"

        # Write YOLO annotation to a file with the same name as the image
        output_file_path = f"/home/user/Documents/Project/yolov5/data/age/labels/test/{image_name.replace('.jpg', '.txt')}"
        with open(output_file_path, 'w') as output_file:
            output_file.write(yolo_annotation)


# Example usage
json_data_path = "/home/user/Documents/Project/yolov5/data/age/test_labels.json"
json_to_yolo(json_data_path)
