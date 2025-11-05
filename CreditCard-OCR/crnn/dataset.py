import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd
import random


class CardDataset(Dataset):
    """
    卡号识别数据集（使用 < 和 > 作为BOS/EOS标记）
    
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
    
    - 标签格式要求：
        ⭐ Excel中的标签必须已经包含 < 和 > 标记 ⭐
        正确格式：'<4532/1234/5678/9012>'
        错误格式：'4532/1234/5678/9012'（缺少<>）
        
        数据集会检查标签格式：
        - 必须以 '<' 开头
        - 必须以 '>' 结尾
        - 不符合格式的样本会被跳过并给出警告
    
    ⭐⭐⭐ 新版：自适应宽度（保持宽高比）⭐⭐⭐
    - 固定高度为 img_height
    - 宽度按原始宽高比自动计算
    - 宽度对齐到4的倍数（适配CNN下采样）
    - 不会变形，更真实
    
    ⭐ 困难样本检测和数据增强 ⭐
    - 困难样本定义：卡号中有任意3个困难数字(2,6,8,9)连续出现
    - 只对困难样本应用数据增强，普通样本不做增强
    """
    
    # ⭐ 字符集：包含 < 和 > 作为BOS/EOS标记
    CHARS = '0123456789/<>'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
    
    # ⭐ 定义BOS和EOS（使用不同的字符）
    BOS_CHAR = '<'
    EOS_CHAR = '>'
    BOS_LABEL = CHAR2LABEL[BOS_CHAR]  # 12
    EOS_LABEL = CHAR2LABEL[EOS_CHAR]  # 13
    
    # ⭐⭐⭐ 定义困难数字集合 ⭐⭐⭐
    HARD_DIGITS = {'2', '6', '8', '9'}

    def __init__(self, image_dir, mode, img_height, img_width):
        """
        Args:
            image_dir: 图片目录路径（train/val/test模式）或 PIL.Image对象（pred模式）
            mode: 'train', 'val', 'test', 'pred'
            img_height: 固定高度
            ⭐ 移除了 img_width 参数 - 宽度自适应 ⭐
        """
        texts = []
        self.mode = mode
        self.image_dir = image_dir
        self.img_height = img_height
        # ⭐ 不再需要 self.img_width ⭐
        
        # ⭐⭐⭐ 新增：困难样本标记列表 ⭐⭐⭐
        self.is_hard_sample = []  # 每个样本是否为困难样本
        
        if mode in ["train", "val", "test"]:
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
        - CardNumberlabel: 文本标签（必须包含<>，如 "<4532/1234/5678/9012>"）
        
        ⭐⭐⭐ 只检查标签格式，不自动修改 ⭐⭐⭐
        - 标签必须以 '<' 开头
        - 标签必须以 '>' 结尾
        - 不符合格式的样本会被跳过并给出警告
        
        ⭐⭐⭐ 新增：困难样本检测 ⭐⭐⭐
        - 检测卡号中有任意3个困难数字(2,6,8,9)连续出现
        - 标记为困难样本，用于后续的数据增强和过采样
        
        仅保留目录中实际存在且为图片的文件。
        """
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
        invalid_format = 0
        
        # ⭐⭐⭐ 新增：困难样本统计 ⭐⭐⭐
        hard_sample_count = 0

        for _, row in df.iterrows():
            name = str(row['filename']).strip()
            label = str(row[label_col]).strip()
            
            # 只检查标签格式，不修改
            # 1. 检查是否以 '<' 开头
            if not label.startswith(self.BOS_CHAR):
                print(f"⚠️ 警告：标签 '{label}' 缺少开头的 '<'（文件：{name}），已跳过")
                invalid_format += 1
                continue
            
            # 2. 检查是否以 '>' 结尾
            if not label.endswith(self.EOS_CHAR):
                print(f"⚠️ 警告：标签 '{label}' 缺少结尾的 '>'（文件：{name}），已跳过")
                invalid_format += 1
                continue
            
            # 3. 验证标签中的所有字符都在字符集中
            invalid_chars = [c for c in label if c not in self.CHARS]
            if invalid_chars:
                print(f"⚠️ 警告：标签 '{label}' 包含无效字符 {set(invalid_chars)}（文件：{name}），已跳过")
                invalid_format += 1
                continue

            # 4. 检查图片文件是否存在
            img_path = os.path.join(self.image_dir, name)
            if not os.path.isfile(img_path):
                missing += 1
                continue
            if not self._is_image_file(name):
                not_image += 1
                continue
            
            # ⭐⭐⭐ 新增：检测困难样本 ⭐⭐⭐
            is_hard = self._is_hard_sample(label)
            
            # 标签格式正确，添加到列表
            filenames.append(name)
            texts.append(label)
            self.is_hard_sample.append(is_hard)  # ⭐ 记录是否为困难样本
            
            if is_hard:
                hard_sample_count += 1

        # 打印统计信息
        print(f"\n{'='*70}")
        print(f"[{self.mode.upper()}] 数据集加载统计")
        print(f"{'='*70}")
        print(f"✅ 成功加载: {len(filenames)} 张图片（来自 {label_filename}）")
        
        # ⭐⭐⭐ 新增：打印困难样本统计 ⭐⭐⭐
        if len(filenames) > 0:
            hard_ratio = hard_sample_count / len(filenames) * 100
            print(f"⭐ 困难样本: {hard_sample_count} 张 ({hard_ratio:.1f}%) - 连续3个困难数字(2,6,8,9)")
        
        if invalid_format > 0:
            print(f"⚠️  格式错误: {invalid_format} 条记录的标签格式不正确（缺少<>或包含无效字符）")
        if missing > 0:
            print(f"⚠️  文件缺失: {missing} 条记录的图片文件未找到")
        if not_image > 0:
            print(f"⚠️  非图片: {not_image} 条记录不是图片文件")
        
        if texts:
            print(f"\n标签格式示例:")
            for i, text in enumerate(texts[:3]):  # 显示前3个标签
                hard_mark = " [困难样本]" if self.is_hard_sample[i] else ""
                print(f"  [{i+1}] '{text}'{hard_mark}")
            print(f"\n✅ 标签格式正确：所有标签都以 '<' 开头，'>' 结尾")
        else:
            print(f"\n❌ 警告：没有加载到任何有效数据！")
            print(f"   请检查：")
            print(f"   1. Excel 中的标签是否包含 '<' 和 '>'")
            print(f"   2. 图片文件是否存在")
        
        print(f"{'='*70}\n")
        
        return filenames, texts

    # ⭐⭐⭐ 新增：困难样本检测方法（连续3个困难数字）⭐⭐⭐
    def _is_hard_sample(self, label):
        """
        判断是否为困难样本
        
        困难样本定义：卡号中有任意3个困难数字(2,6,8,9)连续出现
        
        例如：
        - '<6289/1234/5678/9012>' → 困难样本（628包含3个连续困难数字）
        - '<1268/1234/5678/9012>' → 困难样本（268包含3个连续困难数字）
        - '<1234/5678/9012/3456>' → 普通样本（没有3个连续困难数字）
        - '<6200/1234/5678/9012>' → 普通样本（62后面是0，不连续）
        
        Args:
            label: 标签字符串，如 '<5210/1234/5678/9012>'
        
        Returns:
            bool: True表示困难样本，False表示普通样本
        """
        # 移除BOS和EOS标记，只保留卡号
        card_number = label.strip(self.BOS_CHAR + self.EOS_CHAR)
        
        # 移除斜杠，得到纯数字字符串
        digits_only = card_number.replace('/', '')
        
        # ⭐ 滑动窗口检查：是否有连续3个字符都是困难数字 ⭐
        for i in range(len(digits_only) - 2):  # -2 因为需要检查3个字符
            # 取连续3个字符
            three_chars = digits_only[i:i+3]
            
            # 检查这3个字符是否都是困难数字
            if all(c in self.HARD_DIGITS for c in three_chars):
                return True  # 找到连续3个困难数字
        
        return False  # 没有找到连续3个困难数字

    # ⭐⭐⭐ 新增：数据增强方法（只对困难样本） ⭐⭐⭐
    def _apply_augmentation(self, image, is_hard):
        """
        应用数据增强
        
        ⭐ 只对困难样本（有连续3个困难数字的卡号）应用增强 ⭐
        ⭐ 普通样本不做任何增强，保持原样 ⭐
        
        Args:
            image: PIL.Image对象
            is_hard: 是否为困难样本
        
        Returns:
            PIL.Image: 增强后的图像（或原图）
        """
        if not is_hard:
            # ⭐ 普通样本：不做任何增强，直接返回原图 ⭐
            return image
        
        # ⭐ 困难样本：80%概率应用增强 ⭐
        if random.random() < 0.8:
            # 1. 亮度调整
            if random.random() < 0.5:
                factor = random.uniform(0.6, 1.4)
                image = ImageEnhance.Brightness(image).enhance(factor)
            
            # 2. 对比度调整
            if random.random() < 0.5:
                factor = random.uniform(0.6, 1.4)
                image = ImageEnhance.Contrast(image).enhance(factor)
            
            # 3. 锐度调整
            if random.random() < 0.3:
                factor = random.uniform(0.5, 2.0)
                image = ImageEnhance.Sharpness(image).enhance(factor)
        
        return image

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
            if self.mode in ["train", "val", "test"]:
                file_name = self.file_names[index]
                file_path = os.path.join(self.image_dir, file_name)
                image = Image.open(file_path)
            elif self.mode == "pred":
                image = self.image_dir
            else:
                raise ValueError(f"❌ 不支持的模式: {self.mode}")

            # ⭐⭐⭐ 新增：训练模式下应用数据增强（只对困难样本）⭐⭐⭐
            if self.mode == "train":
                is_hard = self.is_hard_sample[index]
                image = self._apply_augmentation(image, is_hard)

            # ⭐⭐⭐ 图像预处理：固定高度，宽度按比例自适应 ⭐⭐⭐
            image = image.convert('L')
            orig_w, orig_h = image.size
            
            if orig_h == 0 or orig_w == 0:
                raise ValueError(f"❌ 图片尺寸异常: {orig_w}x{orig_h}")
            
            # ⭐ 计算新宽度：保持宽高比 ⭐
            new_w = int(orig_w * self.img_height / orig_h)
            
            # ⭐ 宽度对齐到4的倍数（适配CNN的4倍下采样）⭐
            new_w = (new_w + 3) // 4 * 4
            
            # ⭐ 确保宽度至少为4 ⭐
            new_w = max(4, new_w)
            
            # Resize到新尺寸
            image = image.resize((new_w, self.img_height), Image.BILINEAR)
            image = np.array(image)
            image = image.reshape((1, self.img_height, new_w))
            
            # 归一化到 [-1, 1]
            image = (image / 127.5) - 1.0
            image = torch.FloatTensor(image)

            if len(self.texts) != 0:
                text = self.texts[index]
                
                # ⭐ 将文本转换为label序列（已经包含BOS和EOS）
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
            
            # 返回dummy数据
            dummy_image = torch.full((1, self.img_height, 100), -1.0)
            if len(self.texts) != 0:
                # ⭐ dummy标签：只包含BOS和EOS
                dummy_target = torch.LongTensor([self.BOS_LABEL, self.EOS_LABEL])
                dummy_length = torch.LongTensor([2])
                return dummy_image, dummy_target, dummy_length
            else:
                return dummy_image
def cardnumber_collate_fn(batch):
    """
    ⭐⭐⭐ 新版：支持自适应宽度的collate_fn ⭐⭐⭐
    
    特点：
    - 每张图片宽度不同（保持原始宽高比）
    - padding到batch中的最大宽度
    - 使用黑色填充（-1.0）
    - 配合BOS/EOS标记，提供明确的序列边界
    
    Args:
        batch: [(image, target, target_length), ...]
               其中 image.shape = (1, H, W_i)，每个样本的W_i可能不同
    
    Returns:
        images: (B, 1, H, max_W) - padding后的图像batch
        targets: (sum(target_lengths),) - 拼接的目标序列（包含BOS/EOS）
        target_lengths: (B,) - 每个样本的目标长度
        original_widths: (B,) - 每个样本的原始宽度（用于计算input_lengths）
    """
    images, targets, target_lengths = zip(*batch)
    
    # ⭐ 记录每张图片的原始宽度（padding前）⭐
    original_widths = [img.size(2) for img in images]
    
    # ⭐ 找到batch中的最大宽度 ⭐
    max_w = max(original_widths)
    h = images[0].size(1)  # 高度是固定的
    
    # ⭐ 对每张图片进行padding ⭐
    padded_images = []
    for img in images:
        w = img.size(2)
        if w == max_w:
            # 已经是最大宽度，无需填充
            padded_images.append(img)
        else:
            # ⭐ 用-1.0（黑色）填充右侧 ⭐
            pad = torch.full((1, h, max_w - w), -1.0, dtype=img.dtype)
            padded_img = torch.cat([img, pad], dim=2)
            padded_images.append(padded_img)
    
    # 堆叠成batch
    images = torch.stack(padded_images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    original_widths = torch.LongTensor(original_widths)
    
    return images, targets, target_lengths, original_widths
