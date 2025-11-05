from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import random
import os
import numpy as np

#字体路径
FONT_PATH = "D:\\PayCard_Detection\\Data\\ziti\\Farrington-7B\\Farrington-7B.ttf"
# "D:\PayCard_Detection\Data\ziti\ocr-b-maisfontes.18c2\ocr-b.otf"
#
# FONT_PATH = "D:\\PayCard_Detection\\Data\\ziti\\Times New\\74370-main\\Times New  Roman.TTF\\TIMESBD.TTF"
FONT_SIZE = 50

def group_numbers(numbers, group_sizes):
    grouped = []
    idx = 0
    for size in group_sizes:
        grouped.append(numbers[idx:idx+size])
        idx += size
        if idx >= len(numbers):
            break
    if idx < len(numbers):
        grouped.append(numbers[idx:])
    return ' '.join(grouped)

def add_random_noise(img, amount=0.02):
    """添加椒盐噪声"""
    arr = np.array(img)
    num_salt = np.ceil(amount * arr.size * 0.5)
    num_pepper = np.ceil(amount * arr.size * 0.5)
    # 添加salt
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in arr.shape]
    arr[tuple(coords)] = 255
    # 添加pepper
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in arr.shape]
    arr[tuple(coords)] = 0
    return Image.fromarray(arr)



def random_erase(img, num_rect=1, max_size_ratio=0.2):
    """随机擦除图片的一小块，模拟数字缺失"""
    arr = np.array(img)
    h, w = arr.shape
    for _ in range(num_rect):
        erase_w = random.randint(3, int(w * max_size_ratio))
        erase_h = random.randint(3, int(h * max_size_ratio))
        x = random.randint(0, w - erase_w)
        y = random.randint(0, h - erase_h)
        arr[y:y+erase_h, x:x+erase_w] = 0  # 擦成黑色
    return Image.fromarray(arr)

def generate_card_number_image(numbers, save_path):
    char_width = FONT_SIZE
    img_width = char_width * len(numbers) + 40
    img_height = int(FONT_SIZE * 1.7)
    img = Image.new('L', (img_width, img_height), color=0)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    for i, char in enumerate(numbers):
        x = 10 + i * char_width
        y = 5
        draw.text((x, y), char, fill=255, font=font)
    # img = img.filter(ImageFilter.GaussianBlur(radius=1))
    draw = ImageDraw.Draw(img)
    for i, char in enumerate(numbers):
        x = 10 + i * char_width
        y = 5
        draw.text((x, y), char, fill=255, font=font)
    img = img.filter(ImageFilter.MinFilter(1))
    img = img.filter(ImageFilter.MaxFilter(3))  # 膨胀

    # --- 数据增强部分 ---
    # 随机加噪声
    if random.random() < 0.5:
        img = add_random_noise(img, amount=random.uniform(0.01, 0.03))
    # 随机仿射变形
    # if random.random() < 0.4:
    #     img = random_affine(img)
    #随机擦除部分区域（数字小缺失）
    if random.random() > 0.5:
        img = random_erase(img, num_rect=random.randint(1,2), max_size_ratio=0.15)

    img.save(save_path)

def random_card_number(group_sizes):
    nums = []
    for size in group_sizes:
        nums.append(''.join(str(random.randint(0, 9)) for _ in range(size)))
    return ' '.join(nums)

def generate_dataset(num_images=10, save_dir='dataset'):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(num_images):
        group_patterns = [
            [4,4,4,4],
            [4,6,5],
            [3,4,4,5],
            [4,5,6],
            [5,5,5]
        ]
        group_sizes = random.choice(group_patterns)
        card_number = random_card_number(group_sizes)
        img_name = card_number.replace(' ', '_') + '.png'
        save_path = os.path.join(save_dir, img_name)
        generate_card_number_image(card_number, save_path)

# 生成3000张图片
generate_dataset(10, "D:\\PayCard_Detection\\Data\\test7")