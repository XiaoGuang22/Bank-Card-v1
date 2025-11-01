# hybrid_background_generator_v6_ultimate.py
# 终极版：融合双方优点 + 光照增强 + Excel标签

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
from pathlib import Path
import os
from scipy.ndimage import binary_erosion, binary_dilation
import pandas as pd

class HybridBackgroundGeneratorV6Ultimate:
    """
    混合背景生成器 V6 - 终极版
    
    核心特性：
    1. ⭐ 你的优点：随机擦除、MinFilter+MaxFilter形态学处理 ⭐
    2. ⭐ 我的优点：光照增强（全局/侧光/聚光）、背景多样性 ⭐
    3. ⭐ Excel标签存储（filename | CardNumberlabel）⭐
    4. ⭐ 文件命名：train_1.png, train_2.png, ... ⭐
    """
    
    def __init__(self, font_path: str, real_background_dir: str):
        self.font_path = font_path
        self.real_background_dir = real_background_dir
        
        # ⭐背景混合比例⭐
        self.real_bg_ratio = 0.5
        self.synthetic_bg_ratio = 0.5
        
        # ⭐字体大小配置⭐
        self.font_size_min = 55
        self.font_size_max = 80
        
        # ⭐间距配置⭐
        self.char_spacing_min = 5
        self.char_spacing_max = 12
        self.group_spacing_min = 73
        self.group_spacing_max = 78
        self.margin_left_min = 5
        self.margin_left_max = 10
        self.margin_right_min = 5
        self.margin_right_max = 10
        
        # ⭐背景高度倍数⭐
        self.bg_height_ratio_min = 1.30
        self.bg_height_ratio_max = 1.50
        
        # 卡配置
        self.card_configs = [
            {'name': 'Visa', 'length': 16, 'format': '4-4-4-4', 
             'bin_prefix': ['4'], 'weight': 0.35},
            {'name': 'MasterCard', 'length': 16, 'format': '4-4-4-4',
             'bin_prefix': ['51', '52', '53', '54', '55'], 'weight': 0.15},
            {'name': 'UnionPay_16', 'length': 16, 'format': '4-4-4-4',
             'bin_prefix': ['62'], 'weight': 0.30},
            {'name': 'UnionPay_19', 'length': 19, 'format': '4-4-4-4-3',
             'bin_prefix': ['62'], 'weight': 0.10},
            {'name': 'Amex', 'length': 15, 'format': '4-6-5',
             'bin_prefix': ['34', '37'], 'weight': 0.10},
        ]
        
        # 加载真实背景
        self.real_backgrounds = self._load_real_backgrounds()
        print(f"✓ 加载了 {len(self.real_backgrounds)} 个真实背景")
        
        # 统计
        self.bg_usage_stats = {'real': 0, 'synthetic': 0}
        self.synthetic_type_stats = {'solid_black': 0, 'solid_other': 0, 'gradient': 0, 'perlin': 0, 'texture': 0}
        self.font_size_stats = []
        self.char_height_stats = []
        self.char_spacing_stats = []
        self.group_spacing_stats = []
        self.char_height_ratio_stats = []
        self.bg_size_stats = []
        
        # ⭐光照统计⭐
        self.lighting_stats = {'none': 0, 'global': 0, 'directional': 0, 'spotlight': 0}
        
        # ⭐新增：增强手段统计⭐
        self.augmentation_stats = {
            'random_erase': 0,
            'morphology': 0,
            'emboss': 0,
            'sharpen': 0,
            'blur': 0,
            'noise': 0
        }
        
        # ⭐标签数据⭐
        self.labels_data = []
    
    def _load_real_backgrounds(self):
        """加载真实背景"""
        bg_paths = list(Path(self.real_background_dir).glob('*.jpg')) + \
                   list(Path(self.real_background_dir).glob('*.png')) + \
                   list(Path(self.real_background_dir).glob('*.jpeg'))
        
        if not bg_paths:
            raise ValueError(f"未找到背景图片: {self.real_background_dir}")
        
        backgrounds = []
        size_stats = []
        
        print(f"\n加载真实背景...")
        for bg_path in bg_paths:
            bg = cv2.imread(str(bg_path), cv2.IMREAD_GRAYSCALE)
            
            if bg is None:
                continue
            
            h, w = bg.shape
            size_stats.append((w, h))
            
            backgrounds.append({
                'image': bg,
                'width': w,
                'height': h,
                'name': bg_path.name
            })
        
        if size_stats:
            widths, heights = zip(*size_stats)
            print(f"\n真实背景统计:")
            print(f"  数量: {len(backgrounds)}")
            print(f"  宽度: {min(widths)} - {max(widths)} (平均: {np.mean(widths):.0f})")
            print(f"  高度: {min(heights)} - {max(heights)} (平均: {np.mean(heights):.0f})")
        
        return backgrounds
    
    def calculate_background_size(self, text: str, font_size: int):
        """根据文本和字体大小计算背景尺寸"""
        font = ImageFont.truetype(self.font_path, font_size)
        
        char_spacing = random.randint(self.char_spacing_min, self.char_spacing_max)
        self.char_spacing_stats.append(char_spacing)
        
        group_spacing = random.randint(self.group_spacing_min, self.group_spacing_max)
        self.group_spacing_stats.append(group_spacing)
        
        groups = text.split(' ')
        total_width = 0
        
        for group in groups:
            group_width = 0
            for char in group:
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
                group_width += char_width
            
            if len(group) > 1:
                group_width += (len(group) - 1) * char_spacing
            
            total_width += group_width
        
        if len(groups) > 1:
            total_width += (len(groups) - 1) * group_spacing
        
        margin_left = random.randint(self.margin_left_min, self.margin_left_max)
        margin_right = random.randint(self.margin_right_min, self.margin_right_max)
        bg_width = total_width + margin_left + margin_right
        
        sample_bbox = font.getbbox('0')
        font_height = sample_bbox[3] - sample_bbox[1]
        
        height_ratio = random.uniform(self.bg_height_ratio_min, self.bg_height_ratio_max)
        bg_height = int(font_height * height_ratio)
        
        return bg_width, bg_height, char_spacing, margin_left, group_spacing
    
    def generate_sample(self, save_path: str = None, sample_index: int = None):
        """生成一个训练样本"""
        # 1. 选择卡类型
        config = random.choices(
            self.card_configs,
            weights=[c['weight'] for c in self.card_configs]
        )[0]
        
        # 2. 生成卡号
        card_number = self.generate_luhn_card(config)
        formatted_number = self.format_card_number(card_number, config['format'])
        
        # 3. 确定字体大小
        font_size = random.randint(self.font_size_min, self.font_size_max)
        self.font_size_stats.append(font_size)
        
        # 4. 计算背景尺寸
        img_width, img_height, char_spacing, margin_left, group_spacing = self.calculate_background_size(
            formatted_number, font_size
        )
        
        self.bg_size_stats.append((img_width, img_height))
        
        # 5. 生成图像
        img_array = self.create_image_ultimate(
            formatted_number, 
            img_width, 
            img_height, 
            font_size,
            char_spacing,
            margin_left,
            group_spacing
        )
        
        # 6. 保存图像和标签
        if save_path:
            cv2.imwrite(save_path, img_array)
            
            filename = os.path.basename(save_path)
            label_with_slash = '_' + formatted_number.replace(' ', '/') + '_'
            
            self.labels_data.append({
                'filename': filename,
                'CardNumberlabel': label_with_slash
            })
        
        return img_array, card_number
    
    def create_image_ultimate(self, text, width, height, font_size, char_spacing, margin_left, group_spacing):
        """
        ⭐⭐⭐ 终极版图像生成（融合双方优点） ⭐⭐⭐
        """
        # 1. 获取背景
        background = self.get_background(width, height)
        font = ImageFont.truetype(self.font_path, font_size)
        
        # 2. 渲染文字
        text_mask_array = self._render_text_with_spacing_v5(
            text, font, width, height, char_spacing, margin_left, group_spacing
        )
        
        # 统计字符高度
        text_pixels = np.where(text_mask_array > 128)
        if len(text_pixels[0]) > 0:
            char_height = text_pixels[0].max() - text_pixels[0].min()
            self.char_height_stats.append(char_height)
            
            char_height_ratio = char_height / height
            self.char_height_ratio_stats.append(char_height_ratio)
        
        # ⭐⭐⭐ 3. 形态学处理（采用你的MinFilter+MaxFilter方法）⭐⭐⭐
        if random.random() > 0.5:
            text_mask_array = self._morphology_processing_improved(text_mask_array)
            self.augmentation_stats['morphology'] += 1
        
        # 4. 压印效果
        if random.random() > 0.5:
            text_mask_array = self._add_emboss_effect(text_mask_array)
            self.augmentation_stats['emboss'] += 1
        
        # 5. 合成文字到背景
        result = background.copy()
        text_pixels = text_mask_array > 128
        result[text_pixels] = 255 - background[text_pixels]
        
        # ⭐⭐⭐ 6. 光照增强（我的优点）⭐⭐⭐
        lighting_type = random.choices(
            ['none', 'global', 'directional', 'spotlight'],
            weights=[0.40, 0.25, 0.30, 0.05]
        )[0]
        
        if lighting_type == 'global':
            result = self._apply_lighting_augmentation(result)
            self.lighting_stats['global'] += 1
        elif lighting_type == 'directional':
            result = self._apply_directional_lighting(result)
            self.lighting_stats['directional'] += 1
        elif lighting_type == 'spotlight':
            result = self._apply_spotlight_effect(result)
            self.lighting_stats['spotlight'] += 1
        else:
            self.lighting_stats['none'] += 1
        
        # 7. 边缘锐化
        if random.random() > 0.6:
            result = self._sharpen_edges(result, text_mask_array)
            self.augmentation_stats['sharpen'] += 1
        
        # 8. 高斯模糊
        if random.random() > 0.7:
            kernel_size = random.choice([3, 5])
            result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
            self.augmentation_stats['blur'] += 1
        
        # 9. 高斯噪声
        if random.random() > 0.5:
            noise = np.random.normal(0, random.uniform(2, 5), result.shape)
            result = np.clip(result + noise, 0, 255).astype(np.uint8)
            self.augmentation_stats['noise'] += 1
        
        # ⭐⭐⭐ 10. 随机擦除（你的优点）⭐⭐⭐
        if random.random() > 0.5:
            result = self._random_erase_improved(result)
            self.augmentation_stats['random_erase'] += 1
        
        return result
    
    # ============================================================
    # ⭐⭐⭐ 你的代码优点：形态学处理（MinFilter + MaxFilter）⭐⭐⭐
    # ============================================================
    
    def _morphology_processing_improved(self, text_mask):
        """
        ⭐ 采用你的方法：MinFilter(腐蚀) + MaxFilter(膨胀) ⭐
        
        优点：
        - 简洁高效
        - 模拟字体磨损
        - 比 scipy 的 binary_erosion 更直观
        """
        text_mask_pil = Image.fromarray(text_mask)
        
        # 随机腐蚀强度
        erode_size = random.choice([1, 3])
        text_mask_pil = text_mask_pil.filter(ImageFilter.MinFilter(erode_size))
        
        # 随机膨胀强度
        dilate_size = random.choice([3, 5])
        text_mask_pil = text_mask_pil.filter(ImageFilter.MaxFilter(dilate_size))
        
        return np.array(text_mask_pil)
    
    # ============================================================
    # ⭐⭐⭐ 你的代码优点：随机擦除（Random Erase）⭐⭐⭐
    # ============================================================
    
    def _random_erase_improved(self, img, num_rect=None, max_size_ratio=0.15):
        """
        ⭐ 采用你的随机擦除思路，并改进填充策略 ⭐
        
        模拟：
        - 磨损（黑色填充）
        - 污渍（随机灰度）
        - 高光（白色填充）
        """
        arr = img.copy()
        h, w = arr.shape
        
        # 随机擦除块数
        if num_rect is None:
            num_rect = random.randint(1, 2)
        
        for _ in range(num_rect):
            erase_w = random.randint(3, int(w * max_size_ratio))
            erase_h = random.randint(3, int(h * max_size_ratio))
            x = random.randint(0, w - erase_w)
            y = random.randint(0, h - erase_h)
            
            # ⭐ 改进：随机填充策略 ⭐
            fill_type = random.choices(
                ['black', 'white', 'gray'],
                weights=[0.5, 0.2, 0.3]
            )[0]
            
            if fill_type == 'black':
                fill_value = 0
            elif fill_type == 'white':
                fill_value = 255
            else:
                fill_value = random.randint(50, 200)
            
            arr[y:y+erase_h, x:x+erase_w] = fill_value
        
        return arr
    
    # ============================================================
    # ⭐⭐⭐ 我的代码优点：光照增强 ⭐⭐⭐
    # ============================================================
    
    def _apply_lighting_augmentation(self, img):
        """方案1：全局光照增强"""
        result = img.copy().astype(np.float32)
        
        # 亮度调整
        brightness_factor = random.uniform(0.7, 1.3)
        result = result * brightness_factor
        
        # 对比度调整
        contrast_factor = random.uniform(0.8, 1.2)
        mean = result.mean()
        result = mean + (result - mean) * contrast_factor
        
        # Gamma校正
        if random.random() > 0.5:
            gamma = random.uniform(0.8, 1.2)
            result = 255 * (result / 255) ** gamma
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    def _apply_directional_lighting(self, img):
        """方案2：方向性侧光"""
        h, w = img.shape
        result = img.copy().astype(np.float32)
        
        angle = random.choice([0, 45, 90, 135, 180, 225, 270, 315])
        
        if angle == 0:
            gradient = np.linspace(1.2, 0.8, w)
            lighting_map = np.tile(gradient, (h, 1))
        elif angle == 90:
            gradient = np.linspace(1.2, 0.8, h)
            lighting_map = np.tile(gradient.reshape(-1, 1), (1, w))
        elif angle == 180:
            gradient = np.linspace(0.8, 1.2, w)
            lighting_map = np.tile(gradient, (h, 1))
        elif angle == 270:
            gradient = np.linspace(0.8, 1.2, h)
            lighting_map = np.tile(gradient.reshape(-1, 1), (1, w))
        else:
            x = np.linspace(-1, 1, w)
            y = np.linspace(-1, 1, h)
            X, Y = np.meshgrid(x, y)
            rad = np.radians(angle)
            X_rot = X * np.cos(rad) - Y * np.sin(rad)
            lighting_map = 1.0 + 0.2 * X_rot
            lighting_map = np.clip(lighting_map, 0.8, 1.2)
        
        result = result * lighting_map
        noise = np.random.normal(1.0, 0.02, result.shape)
        result = result * noise
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    def _apply_spotlight_effect(self, img):
        """方案3：聚光灯效果"""
        h, w = img.shape
        result = img.copy().astype(np.float32)
        
        center_x = random.randint(w // 4, 3 * w // 4)
        center_y = random.randint(h // 4, 3 * h // 4)
        
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(w**2 + h**2) / 2
        
        intensity = 1.0 - (distance / max_dist) * random.uniform(0.3, 0.5)
        intensity = np.clip(intensity, 0.7, 1.3)
        
        result = result * intensity
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    # ============================================================
    # 其他辅助方法
    # ============================================================
    
    def _render_text_with_spacing_v5(self, text, font, width, height, char_spacing, margin_left, group_spacing):
        """逐字符渲染"""
        text_mask = Image.new('L', (width, height), color=0)
        draw = ImageDraw.Draw(text_mask)
        
        groups = text.split(' ')
        start_x = margin_left + random.randint(-3, 3)
        
        sample_bbox = font.getbbox('0')
        char_height = sample_bbox[3] - sample_bbox[1]
        start_y = (height - char_height) // 2 + random.randint(-2, 2)
        
        current_x = start_x
        
        for group_idx, group in enumerate(groups):
            for char_idx, char in enumerate(group):
                draw.text((current_x, start_y), char, fill=255, font=font)
                
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
                current_x += char_width
                
                if char_idx < len(group) - 1:
                    current_x += char_spacing
            
            if group_idx < len(groups) - 1:
                current_x += group_spacing
        
        text_mask_array = np.array(text_mask)
        return text_mask_array
    
    def get_background(self, width, height):
        """获取背景"""
        if random.random() < self.real_bg_ratio:
            bg = self._get_real_background(width, height)
            self.bg_usage_stats['real'] += 1
        else:
            bg = self._generate_synthetic_background(width, height)
            self.bg_usage_stats['synthetic'] += 1
        
        return bg
    
    def _get_real_background(self, width, height):
        """获取真实背景"""
        bg_info = random.choice(self.real_backgrounds)
        bg = bg_info['image'].copy()
        bg_h, bg_w = bg.shape
        
        if bg_w >= width and bg_h >= height:
            x_start = random.randint(0, bg_w - width)
            y_start = random.randint(0, bg_h - height)
            crop = bg[y_start:y_start+height, x_start:x_start+width]
        elif bg_w >= width * 0.8 and bg_h >= height * 0.8:
            scale = max(height / bg_h, width / bg_w) * 1.1
            new_w = int(bg_w * scale)
            new_h = int(bg_h * scale)
            bg = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            x_start = random.randint(0, new_w - width)
            y_start = random.randint(0, new_h - height)
            crop = bg[y_start:y_start+height, x_start:x_start+width]
        else:
            crop = cv2.resize(bg, (width, height), interpolation=cv2.INTER_CUBIC)
        
        crop = self._augment_background(crop)
        return crop
    
    def _generate_synthetic_background(self, width, height):
        """生成合成背景"""
        bg_type = random.choices(
            ['solid', 'gradient', 'perlin', 'texture'],
            weights=[0.80, 0.067, 0.067, 0.066]
        )[0]
        
        if bg_type == 'solid':
            bg = self._generate_solid_background(width, height)
        elif bg_type == 'gradient':
            bg = self._generate_gradient_background(width, height)
            self.synthetic_type_stats['gradient'] += 1
        elif bg_type == 'perlin':
            bg = self._generate_perlin_background(width, height)
            self.synthetic_type_stats['perlin'] += 1
        else:
            bg = self._generate_texture_background_v2(width, height)
            self.synthetic_type_stats['texture'] += 1
        
        return bg
    
    def _generate_perlin_background(self, width, height):
        """生成Perlin噪声背景"""
        bg = np.zeros((height, width), dtype=np.float32)
        
        octaves = [(4, 0.5), (8, 0.3), (16, 0.15), (32, 0.05)]
        
        for scale, weight in octaves:
            noise_h = max(height // scale, 2)
            noise_w = max(width // scale, 2)
            
            noise = np.random.randn(noise_h, noise_w)
            noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
            
            bg += noise * weight
        
        bg = (bg - bg.min()) / (bg.max() - bg.min())
        base_gray = random.randint(80, 180)
        contrast = random.uniform(35, 55)
        bg = base_gray + (bg - 0.5) * contrast
        
        bg = np.clip(bg, 0, 255).astype(np.uint8)
        
        if random.random() > 0.5:
            bg = self._add_directional_texture(bg)
        
        return bg
    
    def _add_directional_texture(self, img):
        """添加方向性纹理"""
        h, w = img.shape
        angle = random.choice([0, 45, 90, 135])
        
        if angle == 0:
            pattern = np.tile(np.sin(np.linspace(0, 20*np.pi, w)) * 3, (h, 1))
        elif angle == 90:
            pattern = np.tile(np.sin(np.linspace(0, 20*np.pi, h)).reshape(-1, 1) * 3, (1, w))
        else:
            x = np.linspace(0, 10*np.pi, w)
            y = np.linspace(0, 10*np.pi, h)
            X, Y = np.meshgrid(x, y)
            pattern = np.sin(X + Y) * 3
        
        result = img.astype(np.float32) + pattern
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _generate_texture_background_v2(self, width, height):
        """生成纹理背景"""
        bg = self._generate_perlin_background(width, height)
        
        if random.random() > 0.5:
            center_x = random.randint(width // 4, 3 * width // 4)
            center_y = random.randint(height // 4, 3 * height // 4)
            
            y, x = np.ogrid[:height, :width]
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(width**2 + height**2) / 2
            
            vignette = 1 - (distance / max_dist) * random.uniform(0.2, 0.4)
            vignette = np.clip(vignette, 0.7, 1.0)
            
            bg = (bg * vignette).astype(np.uint8)
        
        return bg
    
    def _generate_solid_background(self, width, height):
        """生成纯色背景"""
        if random.random() < 0.80:
            gray_value = random.randint(0, 30)
            self.synthetic_type_stats['solid_black'] += 1
        else:
            gray_value = random.choice([
                random.randint(20, 60),
                random.randint(80, 120),
                random.randint(140, 180),
                random.randint(200, 235),
            ])
            self.synthetic_type_stats['solid_other'] += 1
        
        bg = np.ones((height, width), dtype=np.uint8) * gray_value
        noise = np.random.normal(0, 3, bg.shape)
        bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
        
        return bg
    
    def _generate_gradient_background(self, width, height):
        """生成渐变背景"""
        direction = random.choice(['horizontal', 'vertical', 'diagonal'])
        
        start_gray = random.randint(30, 120)
        end_gray = random.randint(120, 220)
        
        if direction == 'horizontal':
            gradient = np.linspace(start_gray, end_gray, width)
            bg = np.tile(gradient, (height, 1))
        elif direction == 'vertical':
            gradient = np.linspace(start_gray, end_gray, height)
            bg = np.tile(gradient.reshape(-1, 1), (1, width))
        else:
            x = np.linspace(0, 1, width)
            y = np.linspace(0, 1, height)
            X, Y = np.meshgrid(x, y)
            gradient = (X + Y) / 2
            bg = start_gray + (end_gray - start_gray) * gradient
        
        bg = bg.astype(np.uint8)
        noise = np.random.normal(0, 5, bg.shape)
        bg = np.clip(bg + noise, 0, 255).astype(np.uint8)
        
        return bg
    
    def _augment_background(self, img: np.ndarray) -> np.ndarray:
        """背景数据增强"""
        result = img.copy()
        
        if random.random() > 0.3:
            noise_level = random.uniform(2, 6)
            noise = np.random.normal(0, noise_level, result.shape)
            result = np.clip(result + noise, 0, 255).astype(np.uint8)
        
        if random.random() > 0.5:
            kernel_size = random.choice([3, 5])
            result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
        
        if random.random() > 0.2:
            alpha = random.uniform(0.85, 1.15)
            beta = random.randint(-15, 15)
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        
        if random.random() > 0.8:
            noise_ratio = random.uniform(0.001, 0.005)
            num_salt = int(noise_ratio * result.size * 0.5)
            num_pepper = int(noise_ratio * result.size * 0.5)
            
            coords = [np.random.randint(0, i, num_salt) for i in result.shape]
            result[coords[0], coords[1]] = 255
            
            coords = [np.random.randint(0, i, num_pepper) for i in result.shape]
            result[coords[0], coords[1]] = 0
        
        return result
    
    def _add_emboss_effect(self, text_mask):
        """添加压印效果"""
        text_binary = text_mask > 128
        
        shadow_offset = random.randint(1, 2)
        shadow = np.roll(text_binary, shadow_offset, axis=0)
        shadow = np.roll(shadow, shadow_offset, axis=1)
        
        highlight = np.roll(text_binary, -1, axis=0)
        highlight = np.roll(highlight, -1, axis=1)
        
        result = text_mask.copy().astype(np.float32)
        
        shadow_only = shadow & (~text_binary)
        result[shadow_only] *= 0.7
        
        highlight_only = highlight & (~text_binary)
        result[highlight_only] = np.minimum(result[highlight_only] * 1.3, 255)
        
        return result.astype(np.uint8)
    
    def _sharpen_edges(self, img, text_mask):
        """边缘锐化"""
        edges = cv2.Canny((text_mask > 128).astype(np.uint8) * 255, 50, 150)
        
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        sharpened = cv2.filter2D(img, -1, kernel)
        
        result = img.copy()
        edge_pixels = edges > 0
        
        result[edge_pixels] = (
            img[edge_pixels] * 0.5 + 
            sharpened[edge_pixels] * 0.5
        ).astype(np.uint8)
        
        return result
    
    # ============================================================
    # 数据集生成
    # ============================================================
    
    def generate_dataset(self, num_samples: int, output_dir: str):
        """生成数据集"""
        os.makedirs(output_dir, exist_ok=True)
        
        self.labels_data = []
        
        print(f"\n" + "=" * 70)
        print(f"混合背景生成器 V6 - 终极版")
        print("=" * 70)
        print(f"核心特性:")
        print(f"  1. ⭐ 你的优点：随机擦除 + MinFilter/MaxFilter形态学")
        print(f"  2. ⭐ 我的优点：光照增强（全局40% + 侧光30% + 聚光5%）")
        print(f"  3. ⭐ 字体大小: {self.font_size_min}-{self.font_size_max} pt")
        print(f"  4. ⭐ 背景高度: 字体高度 × {self.bg_height_ratio_min:.2f}-{self.bg_height_ratio_max:.2f}")
        print(f"  5. ⭐ 组间距: {self.group_spacing_min}-{self.group_spacing_max} px")
        print(f"  6. ⭐ 文件命名: train_1.png, train_2.png, ...")
        print(f"  7. ⭐ 标签存储: train_labels.xlsx (filename | CardNumberlabel)")
        print("=" * 70)
        print(f"背景配置:")
        print(f"  真实背景: {len(self.real_backgrounds)} 张")
        print(f"  混合比例: {self.real_bg_ratio*100:.0f}%真实 + {self.synthetic_bg_ratio*100:.0f}%合成")
        print(f"  目标生成数量: {num_samples}")
        print("=" * 70 + "\n")
        
        for i in range(num_samples):
            img_name = f"train_{i+1}.png"
            save_path = os.path.join(output_dir, img_name)
            
            self.generate_sample(save_path, sample_index=i+1)
            
            if (i + 1) % 1000 == 0:
                print(f"  已生成 {i+1}/{num_samples} ({100*(i+1)/num_samples:.1f}%)")
        
        self._save_labels_to_excel(output_dir)
        self._print_statistics(num_samples, output_dir)
    
    def _save_labels_to_excel(self, output_dir: str):
        """保存标签到Excel"""
        if not self.labels_data:
            print("⚠️  警告：没有标签数据需要保存")
            return
        
        df = pd.DataFrame(self.labels_data)
        excel_path = os.path.join(output_dir, 'train_labels.xlsx')
        df.to_excel(excel_path, index=False, engine='openpyxl')
        
        print(f"\n✅ 标签文件已保存:")
        print(f"   路径: {excel_path}")
        print(f"   格式: Excel (.xlsx)")
        print(f"   列名: filename | CardNumberlabel")
        print(f"   行数: {len(df)} 条记录")
        
        print(f"\n📋 标签示例（前5条）:")
        print(df.head().to_string(index=False))
    
    def _print_statistics(self, num_samples: int, output_dir: str):
        """打印统计信息"""
        print("\n" + "=" * 70)
        print("📊 生成统计")
        print("=" * 70)
        
        print("\n【背景使用统计】")
        total = sum(self.bg_usage_stats.values())
        print(f"  真实背景: {self.bg_usage_stats['real']} ({100*self.bg_usage_stats['real']/total:.1f}%)")
        print(f"  合成背景: {self.bg_usage_stats['synthetic']} ({100*self.bg_usage_stats['synthetic']/total:.1f}%)")
        
        if sum(self.synthetic_type_stats.values()) > 0:
            print("\n【合成背景类型统计】")
            synthetic_total = sum(self.synthetic_type_stats.values())
            print(f"  纯黑背景: {self.synthetic_type_stats['solid_black']} ({100*self.synthetic_type_stats['solid_black']/synthetic_total:.1f}%)")
            print(f"  其他纯色: {self.synthetic_type_stats['solid_other']} ({100*self.synthetic_type_stats['solid_other']/synthetic_total:.1f}%)")
            print(f"  渐变背景: {self.synthetic_type_stats['gradient']} ({100*self.synthetic_type_stats['gradient']/synthetic_total:.1f}%)")
            print(f"  Perlin背景: {self.synthetic_type_stats['perlin']} ({100*self.synthetic_type_stats['perlin']/synthetic_total:.1f}%)")
            print(f"  纹理背景: {self.synthetic_type_stats['texture']} ({100*self.synthetic_type_stats['texture']/synthetic_total:.1f}%)")
            
            actual_black_ratio = self.synthetic_type_stats['solid_black'] / total * 100
            print(f"\n  ⭐ 实际纯黑背景占总体: {actual_black_ratio:.1f}% (目标40%)")
        
        # ⭐⭐⭐ 光照增强统计 ⭐⭐⭐
        if sum(self.lighting_stats.values()) > 0:
            print("\n【光照增强统计】")
            lighting_total = sum(self.lighting_stats.values())
            print(f"  无处理: {self.lighting_stats['none']} ({100*self.lighting_stats['none']/lighting_total:.1f}%)")
            print(f"  全局光照: {self.lighting_stats['global']} ({100*self.lighting_stats['global']/lighting_total:.1f}%)")
            print(f"  方向性侧光: {self.lighting_stats['directional']} ({100*self.lighting_stats['directional']/lighting_total:.1f}%)")
            print(f"  聚光灯效果: {self.lighting_stats['spotlight']} ({100*self.lighting_stats['spotlight']/lighting_total:.1f}%)")
        
        # ⭐⭐⭐ 新增：增强手段统计 ⭐⭐⭐
        if sum(self.augmentation_stats.values()) > 0:
            print("\n【数据增强手段统计】")
            print(f"  随机擦除: {self.augmentation_stats['random_erase']} ({100*self.augmentation_stats['random_erase']/total:.1f}%)")
            print(f"  形态学处理: {self.augmentation_stats['morphology']} ({100*self.augmentation_stats['morphology']/total:.1f}%)")
            print(f"  压印效果: {self.augmentation_stats['emboss']} ({100*self.augmentation_stats['emboss']/total:.1f}%)")
            print(f"  边缘锐化: {self.augmentation_stats['sharpen']} ({100*self.augmentation_stats['sharpen']/total:.1f}%)")
            print(f"  高斯模糊: {self.augmentation_stats['blur']} ({100*self.augmentation_stats['blur']/total:.1f}%)")
            print(f"  高斯噪声: {self.augmentation_stats['noise']} ({100*self.augmentation_stats['noise']/total:.1f}%)")
        
        if self.font_size_stats:
            print("\n【字体大小统计】")
            print(f"  配置范围: {self.font_size_min}-{self.font_size_max} pt")
            print(f"  实际范围: {min(self.font_size_stats)}-{max(self.font_size_stats)} pt")
            print(f"  平均值: {np.mean(self.font_size_stats):.1f} pt")
            print(f"  中位数: {np.median(self.font_size_stats):.1f} pt")
        
        if self.bg_size_stats:
            print("\n【背景尺寸统计】")
            widths, heights = zip(*self.bg_size_stats)
            print(f"  宽度范围: {min(widths)}-{max(widths)} px (平均: {np.mean(widths):.0f})")
            print(f"  高度范围: {min(heights)}-{max(heights)} px (平均: {np.mean(heights):.0f})")
            
            aspect_ratios = [w/h for w, h in self.bg_size_stats]
            print(f"  宽高比: {min(aspect_ratios):.1f}-{max(aspect_ratios):.1f} (平均: {np.mean(aspect_ratios):.1f})")
        
        if self.char_height_stats:
            print("\n【字符高度统计】")
            print(f"  实际范围: {min(self.char_height_stats)}-{max(self.char_height_stats)} px")
            print(f"  平均值: {np.mean(self.char_height_stats):.1f} px")
            print(f"  中位数: {np.median(self.char_height_stats):.1f} px")
        
        if self.char_height_ratio_stats:
            print("\n【字符高度占比统计】")
            ratios_pct = [r * 100 for r in self.char_height_ratio_stats]
            print(f"  实际范围: {min(ratios_pct):.1f}% - {max(ratios_pct):.1f}%")
            print(f"  平均值: {np.mean(ratios_pct):.1f}%")
            print(f"  中位数: {np.median(ratios_pct):.1f}%")
        
        if self.char_spacing_stats:
            print("\n【字符间距统计】")
            print(f"  配置范围: {self.char_spacing_min}-{self.char_spacing_max} px")
            print(f"  实际范围: {min(self.char_spacing_stats)}-{max(self.char_spacing_stats)} px")
            print(f"  平均值: {np.mean(self.char_spacing_stats):.1f} px")
            print(f"  中位数: {np.median(self.char_spacing_stats):.1f} px")
        
        if self.group_spacing_stats:
            print("\n【组间距统计】")
            print(f"  配置范围: {self.group_spacing_min}-{self.group_spacing_max} px")
            print(f"  实际范围: {min(self.group_spacing_stats)}-{max(self.group_spacing_stats)} px")
            print(f"  平均值: {np.mean(self.group_spacing_stats):.1f} px")
            print(f"  中位数: {np.median(self.group_spacing_stats):.1f} px")
        
        print("=" * 70)
        
        print("\n✓ 数据集生成完成！")
        print(f"  保存位置: {output_dir}")
        print(f"  图像数量: {num_samples}")
        print(f"  标签文件: train_labels.xlsx")
    
    # ============================================================
    # 辅助方法
    # ============================================================
    
    def generate_luhn_card(self, config):
        """生成符合Luhn算法的卡号"""
        length = config['length']
        bin_prefix = random.choice(config['bin_prefix'])
        
        card_number = bin_prefix
        while len(card_number) < length - 1:
            card_number += str(random.randint(0, 9))
        
        check_digit = self.calculate_luhn_checksum(card_number)
        card_number += str(check_digit)
        
        return card_number
    
    def calculate_luhn_checksum(self, card_number):
        """计算Luhn校验位"""
        digits = [int(d) for d in card_number]
        
        for i in range(len(digits) - 1, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        
        total = sum(digits)
        check_digit = (10 - (total % 10)) % 10
        
        return check_digit
    
    def format_card_number(self, card_number, format_str):
        """格式化卡号"""
        parts = format_str.split('-')
        formatted = []
        pos = 0
        
        for part_len in parts:
            part_len = int(part_len)
            formatted.append(card_number[pos:pos+part_len])
            pos += part_len
        
        return ' '.join(formatted)


# ============================================================
# 使用示例
# ============================================================
if __name__ == '__main__':
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    generator = HybridBackgroundGeneratorV6Ultimate(
        font_path=os.path.join(BASE_DIR, '../Font/Farrington-7B.ttf'),
        real_background_dir=os.path.join(BASE_DIR, '../data/true_background')
    )
    
    # ⭐ 先生成100张测试 ⭐
    output_dir = os.path.join(os.path.dirname(BASE_DIR), 'datasets/train')
    generator.generate_dataset(10000, output_dir)
    
    # ⭐ 如果效果好，取消注释生成完整数据集 ⭐
    # output_dir = os.path.join(os.path.dirname(BASE_DIR), 'datasets/train_ultimate')
    # generator.generate_dataset(10000, output_dir)
