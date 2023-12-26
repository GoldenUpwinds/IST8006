from PIL import Image, ImageDraw

def draw_bbox(image_path, bbox):
    # 打开图像
    image = Image.open(image_path)

    # 创建一个可绘制的对象
    draw = ImageDraw.Draw(image)

    # 提取边界框坐标
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

    # 绘制矩形
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    # 显示图像
    image.show()

# 你需要提供图像路径和边界框信息
image_path = "/home/user/Documents/Project/yolov5/data/age/images/test/test0.jpg"
bbox = {
    "x1": 182.485987981248,
    "y1": 146.108790384998,
    "x2": 247.364943654497,
    "y2": 210.987746058247
}

# 绘制并显示图像
draw_bbox(image_path, bbox)
