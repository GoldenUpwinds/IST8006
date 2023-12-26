import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from torchvision import transforms
from PIL import Image
from agemodels import UNet as Net  # Assuming your AgeNet model is in agemodels module
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


# 定义评估函数
def evaluate(model, image_folder, label_folder, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_loss = 0.0
    predictions = []
    ground_truth = []

    # 获取图像和标签文件列表
    image_files = sorted(os.listdir(image_folder))
    label_files = sorted(os.listdir(label_folder))

    # 确保图像和标签文件一一对应
    assert len(image_files) == len(label_files), "Number of images and labels must be the same."

    # 初始化不同误差范围内的计数器
    count_05 = count_15 = count_5 = count_10 = 0

    for i in tqdm(range(len(image_files)), desc="Evaluating"):
        image_path = os.path.join(image_folder, image_files[i])
        label_path = os.path.join(label_folder, label_files[i])

        # 读取图像和标签
        image = Image.open(image_path).convert('RGB')
        label = torch.tensor(int(open(label_path).read().strip()), dtype=torch.float32).unsqueeze(0)

        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension

        image, label = image.to(device), label.to(device)

        # 模型推断
        outputs = torch.round(model(image))
        label = label.squeeze()

        # 计算损失
        loss = criterion(outputs.squeeze(), label.float())
        total_loss += loss.item()

        # 保存预测结果和真实标签
        prediction_value = outputs.item()
        predictions.append(prediction_value)
        ground_truth.extend([label.item()])

        # 统计不同误差范围内的样本数量
        absolute_error = abs(label.item() - outputs.item())
        if absolute_error < 0.5:
            count_05 += 1
        if absolute_error < 1.5:
            count_15 += 1
        if absolute_error < 5:
            count_5 += 1
        if absolute_error < 10:
            count_10 += 1

    print(predictions)
    avg_loss = total_loss / len(image_files)
    mae = mean_absolute_error(ground_truth, predictions)

    # 计算比例
    total_samples = len(image_files)
    ratio_05 = count_05 / total_samples
    ratio_15 = count_15 / total_samples
    ratio_5 = count_5 / total_samples
    ratio_10 = count_10 / total_samples

    print(f"Evaluation Loss: {avg_loss}, Mean Absolute Error: {mae}")
    print(f"比例（误差 < 0.5）: {ratio_05:.2%}")
    print(f"比例（误差 < 1.5）: {ratio_15:.2%}")
    print(f"比例（误差 < 5）: {ratio_5:.2%}")
    print(f"比例（误差 < 10）: {ratio_10:.2%}")


# 设置数据路径
val_image_folder = os.path.join(ROOT, 'data', 'age', 'images', 'test_filtered')
val_label_folder = os.path.join(ROOT, 'data', 'age', 'crop_labels_filtered', 'test')

# 初始化网络和损失函数
model = Net()
criterion = nn.L1Loss()

# 加载之前保存的最佳模型参数
model.load_state_dict(torch.load(os.path.join(ROOT, 'data', 'age', 'output_unet', 'best_model.pth')))

# 评估模型
evaluate(model, val_image_folder, val_label_folder, criterion)
