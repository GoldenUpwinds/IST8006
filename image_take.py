import cv2
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
# 读取JSON文件
json_file_name = 'test_labels.json'
json_file_path = os.path.join(ROOT, 'data', 'age', json_file_name)
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 遍历每个标签
for label_data in data:
    image_name = label_data["name"]
    image_type = 'test'
    image_path = os.path.join(ROOT, 'data', 'age', 'images', image_type, image_name)
    image = cv2.imread(str(image_path))

    # 提取bbox信息
    x1 = int(label_data["bbox"]["x1"])
    y1 = int(label_data["bbox"]["y1"])
    x2 = int(label_data["bbox"]["x2"])
    y2 = int(label_data["bbox"]["y2"])

    # 截取图像
    cropped_image = image[y1:y2, x1:x2]

    # 调整大小为512x512
    resized_image = cv2.resize(cropped_image, (128, 128), interpolation=cv2.INTER_LANCZOS4)

    # 保存图像
    output_path = os.path.join(ROOT, 'data', 'age', 'crop_images', image_type, label_data["name"])
    cv2.imwrite(str(output_path), resized_image)
