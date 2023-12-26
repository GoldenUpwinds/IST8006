import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# 读取 JSON 文件
json_file_name = 'test_labels.json'
json_file_path = os.path.join(ROOT, 'data', 'age', json_file_name)
with open(json_file_path, 'r') as json_file:
    labels_data = json.load(json_file)

# 获取当前脚本所在的文件夹作为根目录
ROOT = os.path.dirname(os.path.abspath(__file__))

# 构建输出目录
label_type = 'test'
output_dir = os.path.join(ROOT, 'data', 'age', 'crop_labels', label_type)
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录，如果不存在的话

# 遍历每个标签
for label_data in labels_data:
    # 获取图像名称和对应的 age
    image_name = label_data["name"]
    age = label_data["age"]

    # 构建输出文件路径
    output_file_path = os.path.join(output_dir, f"{image_name.split('.')[0]}.txt")

    # 将 age 写入文件
    with open(output_file_path, 'w') as output_file:
        output_file.write(str(age))
