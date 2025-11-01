import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd


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
    
    - 固定高度，宽度按比例缩放
    - 使用方案A：统一黑色填充（-1.0）
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
        self.img_width = img_width
        
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
        invalid_format = 0  # ⭐ 统计格式错误的数量

        for _, row in df.iterrows():
            name = str(row['filename']).strip()
            label = str(row[label_col]).strip()
            
            # ⭐⭐⭐ 只检查标签格式，不修改 ⭐⭐⭐
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
            
            # ⭐ 标签格式正确，添加到列表
            filenames.append(name)
            texts.append(label)

        # 打印统计信息
        print(f"\n{'='*70}")
        print(f"[{self.mode.upper()}] 数据集加载统计")
        print(f"{'='*70}")
        print(f"✅ 成功加载: {len(filenames)} 张图片（来自 {label_filename}）")
        
        if invalid_format > 0:
            print(f"⚠️  格式错误: {invalid_format} 条记录的标签格式不正确（缺少<>或包含无效字符）")
        if missing > 0:
            print(f"⚠️  文件缺失: {missing} 条记录的图片文件未找到")
        if not_image > 0:
            print(f"⚠️  非图片: {not_image} 条记录不是图片文件")
        
        if texts:
            print(f"\n标签格式示例:")
            for i, text in enumerate(texts[:3]):  # 显示前3个标签
                print(f"  [{i+1}] '{text}'")
            print(f"\n✅ 标签格式正确：所有标签都以 '<' 开头，'>' 结尾")
        else:
            print(f"\n❌ 警告：没有加载到任何有效数据！")
            print(f"   请检查：")
            print(f"   1. Excel 中的标签是否包含 '<' 和 '>'")
            print(f"   2. 图片文件是否存在")
        
        print(f"{'='*70}\n")
        
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
            if self.mode in ["train", "val", "test"]:
                file_name = self.file_names[index]
                file_path = os.path.join(self.image_dir, file_name)
                image = Image.open(file_path)
            elif self.mode == "pred":
                image = self.image_dir
            else:
                raise ValueError(f"❌ 不支持的模式: {self.mode}")

            # 图像预处理：固定高度，宽度按比例缩放
            image = image.convert('L')
            orig_w, orig_h = image.size
            
            if orig_h == 0 or orig_w == 0:
                raise ValueError(f"❌ 图片尺寸异常: {orig_w}x{orig_h}")
            
            new_w = max(4, int(round(orig_w * (self.img_height / float(orig_h)))))
            
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


# ========== 方案A：统一黑色填充（与BOS/EOS标记兼容）==========
def cardnumber_collate_fn(batch):
    """
    方案A：统一黑色填充（-1.0）+ BOS/EOS标记
    
    特点：
    - 训练测试完全一致
    - 模型会学习"黑色边缘=结束"
    - 配合BOS/EOS标记，提供明确的序列边界
    - 简单高效
    
    Args:
        batch: [(image, target, target_length), ...]
    
    Returns:
        images: (B, 1, H, W) - 填充后的图像batch
        targets: (sum(target_lengths),) - 拼接的目标序列（包含BOS/EOS）
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


# ========== 测试代码 ==========
if __name__ == '__main__':
    print("="*70)
    print("测试 CardDataset（BOS='<', EOS='>'）")
    print("="*70)
    
    # 打印字符集信息
    print(f"\n字符集: '{CardDataset.CHARS}'")
    print(f"字符到标签的映射:")
    for char, label in sorted(CardDataset.CHAR2LABEL.items(), key=lambda x: x[1]):
        print(f"  '{char}' -> {label}")
    
    print(f"\nBOS标记: '{CardDataset.BOS_CHAR}' (label={CardDataset.BOS_LABEL})")
    print(f"EOS标记: '{CardDataset.EOS_CHAR}' (label={CardDataset.EOS_LABEL})")
    
    # 测试标签格式检查
    print(f"\n{'='*70}")
    print("测试标签格式检查")
    print(f"{'='*70}")
    
    test_cases = [
        ('<1234567890>', True, '正确格式'),
        ('<5210/0000/1772/3664>', True, '正确格式（带斜杠）'),
        ('1234567890', False, '缺少<>'),
        ('<1234567890', False, '缺少>'),
        ('1234567890>', False, '缺少<'),
        ('<1234/5678/9012>', True, '正确格式'),
        ('<1234_5678>', False, '包含无效字符_'),
    ]
    
    for label, should_pass, desc in test_cases:
        # 检查格式
        valid = True
        reason = []
        
        if not label.startswith(CardDataset.BOS_CHAR):
            valid = False
            reason.append("缺少开头的'<'")
        
        if not label.endswith(CardDataset.EOS_CHAR):
            valid = False
            reason.append("缺少结尾的'>'")
        
        invalid_chars = [c for c in label if c not in CardDataset.CHARS]
        if invalid_chars:
            valid = False
            reason.append(f"包含无效字符{set(invalid_chars)}")
        
        status = "✅" if valid == should_pass else "❌"
        print(f"\n{status} 标签: '{label}'")
        print(f"   描述: {desc}")
        print(f"   预期: {'通过' if should_pass else '不通过'}")
        print(f"   实际: {'通过' if valid else '不通过'}")
        if not valid:
            print(f"   原因: {', '.join(reason)}")
    
    print(f"\n{'='*70}")
    print("✅ 测试完成！")
    print(f"{'='*70}")
