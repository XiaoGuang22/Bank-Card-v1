# 数据生成和划分指南

## 概述

修改后的数据生成器现在使用 `/` 作为卡号分组符，与字符集 `'0123456789/'` 一致。

## 修改内容

1. **format_card_number**: 返回格式从空格分隔改为 `/` 分隔
2. **_render_text_with_spacing_v5**: 渲染时使用 `/` 分组
3. **标签保存**: 保存带格式的卡号（如 `4550/7033/5915/4387`）而非连续数字

## 数据生成步骤

### 1. 生成合成数据

```python
from Utils.random_dataset import HybridBackgroundGeneratorV5
import os

# 初始化生成器
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
generator = HybridBackgroundGeneratorV5(
    font_path=os.path.join(BASE_DIR, '../Font/Farrington-7B.ttf'),
    real_background_dir=os.path.join(BASE_DIR, '../data/true_background')
)

# 生成数据集（例如10000张）
output_dir = os.path.join(os.path.dirname(BASE_DIR), 'datasets')
generator.generate_dataset(10000, output_dir)
```

生成的数据将保存在 `datasets/` 目录下，包含：
- 图片文件: `train_1.png`, `train_2.png`, ...
- 标签文件: `labels.xlsx`

### 2. 划分训练集和验证集

使用提供的划分脚本：

```bash
cd CreditCard-OCR/scripts
python split_dataset.py ../datasets 0.8
```

参数说明：
- 第一个参数：源数据目录（包含images和labels.xlsx）
- 第二个参数：训练集比例（0.8表示80%训练，20%验证）

运行后会在 `datasets/` 下创建：
- `train/`: 训练集（8000张）
  - `train_1.png`, `train_2.png`, ...
  - `labels.xlsx`
- `val/`: 验证集（2000张）
  - `val_1.png`, `val_2.png`, ...
  - `labels.xlsx`

### 3. 准备测试集

将28张真实数据放在 `datasets/test/` 目录下：
- 图片文件
- `labels.xlsx` (标签格式也应使用 `/` 分隔)

## 标签格式示例

现在所有标签都应该使用 `/` 分隔：

| filename | CardNumberlabel |
|----------|----------------|
| train_1.png | 4550/7033/5915/4387 |
| train_2.png | 6225/1234/5678/9012 |
| train_3.png | 3782/822463/10005 |

## 注意事项

1. **GUI显示**: GUI会自动过滤掉 `/` 符号，只显示纯数字
2. **向后兼容**: 旧的连续数字格式标签仍然可以工作（会被自动忽略掉）
3. **重新生成**: 如果之前已经生成了数据，需要重新生成才能体现这些修改

## 训练

数据准备好后，直接运行训练脚本：

```bash
cd CreditCard-OCR/crnn
python train.py
```

模型将学习识别带 `/` 的卡号格式，并能正确区分组间间距。

