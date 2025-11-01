import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd  # 读取 Excel 标签


class CardDataset(Dataset):
    """
    卡号识别数据集
    - 数据目录结构：
        datasets/
          ├─ train/
          │   ├─ images...
          │   └─ train_labels.xlsx
          ├─ val/
          │   ├─ images...
          │   └─ val_labels.xlsx
          └─ test/
              ├─ images...
              └─ test_labels.xlsx
    - 固定高度，宽度按比例缩放
    - 使用方案A：统一黑色填充（-1.0）
    """
    CHARS = '0123456789/'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, image_dir, mode, img_height, img_width=None):
        """
        Args:
            image_dir: 图片目录路径（train/val/test模式）或 PIL.Image对象（pred模式）
            mode: 'train', 'val', 'test', 'pred'
            img_height: 固定高度
            img_width: 保留参数（兼容旧代码），实际不使用
        """
        texts = []
        self.mode = mode
        self.image_dir = image_dir
        self.img_height = img_height
        self.img_width = img_width  # 保留但不使用
        
        if mode in ["train", "val", "test"]:  # ✅ 修改：添加 test 模式
            file_names, texts = self._load_from_labels_excel()
            self.file_names = file_names
        self.texts = texts

    def _load_from_labels_excel(self):
        """
        从 Excel 读取标签：根据 mode 自动选择标签文件
        - train 模式 → train_labels.xlsx
        - val 模式 → val_labels.xlsx
        - test 模式 → test_labels.xlsx
        
        Excel 格式要求：
        - filename: 图片文件名
        - CardNumberlabel: 文本标签
        
        仅保留目录中实际存在且为图片的文件。
        """
        # ✅ 修改：根据 mode 确定标签文件名
        label_filename_map = {
            'train': 'train_labels.xlsx',
            'val': 'val_labels.xlsx',
            'test': 'test_labels.xlsx'
        }
        
        label_filename = label_filename_map.get(self.mode)
        if label_filename is None:
            print(f"❌ 错误：不支持的模式 '{self.mode}'")
            return [], []
        
        excel_path = os.path.join(self.image_dir, label_filename)
        
        # ✅ 修改：更详细的错误提示
        if not os.path.isfile(excel_path):
            print(f"❌ 错误：未找到标签文件 {excel_path}")
            print(f"   提示：请确保 {label_filename} 存在于 {self.image_dir} 目录下")
            return [], []

        try:
            df = pd.read_excel(excel_path, engine='openpyxl')
        except Exception as e:
            print(f"❌ 错误：无法读取 Excel {excel_path}")
            print(f"   详细信息：{e}")
            return [], []

        # 兼容不同列名
        if 'filename' not in df.columns:
            print(f"❌ 错误：Excel 缺少列 'filename'")
            print(f"   当前列名：{list(df.columns)}")
            return [], []
        
        label_col = 'CardNumberlabel' if 'CardNumberlabel' in df.columns else ('label' if 'label' in df.columns else None)
        if label_col is None:
            print(f"❌ 错误：Excel 缺少列 'CardNumberlabel'（或备用列 'label'）")
            print(f"   当前列名：{list(df.columns)}")
            return [], []

        filenames = []
        texts = []
        missing = 0
        not_image = 0

        for _, row in df.iterrows():
            name = str(row['filename']).strip()
            label = str(row[label_col]).strip()

            img_path = os.path.join(self.image_dir, name)
            if not os.path.isfile(img_path):
                missing += 1
                continue
            if not self._is_image_file(name):
                not_image += 1
                continue
            filenames.append(name)
            texts.append(label)

        if missing > 0:
            print(f"⚠️ 提示：{missing} 条记录的图片文件未找到，已跳过")
        if not_image > 0:
            print(f"⚠️ 提示：{not_image} 条记录不是图片文件，已跳过")

        # ✅ 修改：更详细的加载信息
        print(f"✅ [{self.mode.upper()}] 加载完成：共 {len(filenames)} 张图片（来自 {label_filename}）")
        return filenames, texts
    
    def _is_image_file(self, filename):
        """检查是否为图片文件"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        return os.path.splitext(filename.lower())[1] in valid_extensions

    def __len__(self):
        if self.mode == "pred":
            return 1
        else:
            return len(self.file_names)
        
    def __getitem__(self, index):
        try:
            if self.mode in ["train", "val", "test"]:  # ✅ 修改：添加 test 模式
                file_name = self.file_names[index]
                file_path = os.path.join(self.image_dir, file_name)
                image = Image.open(file_path)
            elif self.mode == "pred":
                # 此时image_dir为PIL.Image对象
                image = self.image_dir
            else:
                raise ValueError(f"❌ 不支持的模式: {self.mode}")

            # 图像预处理：固定高度，宽度按比例缩放
            image = image.convert('L')
            orig_w, orig_h = image.size
            
            # ✅ 修复：处理异常尺寸
            if orig_h == 0 or orig_w == 0:
                raise ValueError(f"❌ 图片尺寸异常: {orig_w}x{orig_h}")
            
            # 按比例缩放宽度
            new_w = max(4, int(round(orig_w * (self.img_height / float(orig_h)))))
            
            # 调整大小
            image = image.resize((new_w, self.img_height), Image.BILINEAR)
            image = np.array(image)
            image = image.reshape((1, self.img_height, new_w))
            
            # 归一化到[-1, 1]
            image = (image / 127.5) - 1.0
            image = torch.FloatTensor(image)

            # 返回数据
            if len(self.texts) != 0:
                text = self.texts[index]
                target = [self.CHAR2LABEL[c] for c in text]
                target_length = [len(target)]
                target = torch.LongTensor(target)
                target_length = torch.LongTensor(target_length)
                return image, target, target_length
            else:
                return image
                
        except Exception as e:
            print(f"❌ 错误：处理图片时出错 (index={index})")
            if self.mode in ["train", "val", "test"]:
                print(f"   文件名：{self.file_names[index]}")
            print(f"   详细信息：{e}")
            # 返回一个默认的空白图片
            dummy_image = torch.full((1, self.img_height, 100), -1.0)
            if len(self.texts) != 0:
                dummy_target = torch.LongTensor([1])  # 默认标签
                dummy_length = torch.LongTensor([1])
                return dummy_image, dummy_target, dummy_length
            else:
                return dummy_image


# ========== 方案A：统一黑色填充 ==========
def cardnumber_collate_fn(batch):
    """
    方案A：统一黑色填充（-1.0）
    
    特点：
    - 训练测试完全一致
    - 模型会学习"黑色边缘=结束"
    - 简单高效
    
    Args:
        batch: [(image, target, target_length), ...]
    
    Returns:
        images: (B, 1, H, W) - 填充后的图像batch
        targets: (sum(target_lengths),) - 拼接的目标序列
        target_lengths: (B,) - 每个样本的目标长度
        original_widths: (B,) - 每个样本的原始宽度（用于计算input_lengths）
    """
    images, targets, target_lengths = zip(*batch)
    
    # 记录原始宽度（填充前的宽度）
    original_widths = [img.size(2) for img in images]
    
    # 找到batch中的最大宽度
    max_w = max(original_widths)
    h = images[0].size(1)
    
    padded_images = []
    for img in images:
        w = img.size(2)
        if w == max_w:
            # 已经是最大宽度，无需填充
            padded_images.append(img)
        else:
            # ⭐ 方案A：用-1.0（黑色）填充右侧 ⭐
            pad = torch.full((1, h, max_w - w), -1.0, dtype=img.dtype)
            padded_images.append(torch.cat([img, pad], dim=2))
    
    # 堆叠成batch
    images = torch.stack(padded_images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    original_widths = torch.LongTensor(original_widths)
    
    return images, targets, target_lengths, original_widths
