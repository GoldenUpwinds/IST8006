import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from agemodels import AgeNet as Net

ROOT = os.path.dirname(os.path.abspath(__file__))


# 定义自定义数据集类
class AgeDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform

        # 获取图像和标签文件列表
        image_files = sorted(os.listdir(image_folder))
        label_files = sorted(os.listdir(label_folder))

        # 确保图像和标签文件一一对应
        assert len(image_files) == len(label_files), "Number of images and labels must be the same."

        self.image_paths = [os.path.join(image_folder, img_file) for img_file in image_files]
        self.label_paths = [os.path.join(label_folder, label_file) for label_file in label_files]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path).convert('RGB')
        label = torch.tensor(int(open(label_path).read().strip()), dtype=torch.float32).unsqueeze(0)  # 将标签转为张量并添加一个维度

        if self.transform:
            image = self.transform(image)

        return image, label


# 定义训练函数
def train(model, dataloader, criterion, optimizer, num_epochs=10,
          save_dir=os.path.join(ROOT, 'data', 'age', 'saved_models'),
          plot_path=os.path.join(ROOT, 'data', 'age', 'loss_plot.png')):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    losses = []
    best_loss = float('inf')  # 初始化最小损失为正无穷
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            labels = labels.squeeze()

            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

        # 比较当前损失和最小损失，如果更小则保存为最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_save_path)
            print(f"Best model saved at: {best_model_save_path}")

        # 在循环中的适当位置添加以下代码，替代原有的保存上一个 epoch 结束时的模型的代码
        if (epoch + 1) % 50 == 0:
            # 每50个epoch保存一次模型，并以当前epoch数命名
            model_save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at: {model_save_path}")

    # 绘制折线图并保存
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()

    # 保存折线图到指定文件夹
    plt.savefig(plot_path)


# 设置数据增强和数据加载器
transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),  # 随机旋转
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.ColorJitter(brightness=0.2,  # 随机颜色调整
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.2),
    transforms.RandomResizedCrop(size=(128, 128),  # 随机裁剪和缩放
                                 scale=(0.8, 1.0),
                                 ratio=(0.8, 1.2)),
    transforms.RandomAffine(degrees=15,  # 随机仿射变换
                            translate=(0.1, 0.1),
                            scale=(0.8, 1.2)),
    transforms.ToTensor(),  # 转换为张量
])

dataset = AgeDataset(image_folder=os.path.join(ROOT, 'data', 'age', 'crop_images', 'train'),
                     label_folder=os.path.join(ROOT, 'data', 'age', 'crop_labels', 'train'), transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# 初始化网络、损失函数和优化器
model = Net()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 训练模型
train(model, dataloader, criterion, optimizer, num_epochs=1000)
