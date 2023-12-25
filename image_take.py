import cv2
import json
import numpy as np

# 读取JSON文件
json_file_path = '/home/user/Documents/Project/yolov5/data/age/test_labels.json'  # 替换成你的JSON文件路径
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 遍历每个标签
for label_data in data:
    image_name = label_data["name"]
    image_path = f"/home/user/Documents/Project/yolov5/data/age/images/test/{image_name}"  # Update this path based on your image location
    image = cv2.imread(image_path)

    # 提取bbox信息
    x1 = int(label_data["bbox"]["x1"])
    y1 = int(label_data["bbox"]["y1"])
    x2 = int(label_data["bbox"]["x2"])
    y2 = int(label_data["bbox"]["y2"])

    # 截取图像
    cropped_image = image[y1:y2, x1:x2]

    # 调整大小为512x512
    resized_image = cv2.resize(cropped_image, (512, 512))

    # 保存图像
    output_path = f'/home/user/Documents/Project/yolov5/data/age/crop_images/test/{label_data["name"]}'  # 替换成你想保存的路径
    cv2.imwrite(output_path, resized_image)
