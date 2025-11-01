# Bank-Card-v1

银行卡识别系统（毕业设计）

## 项目简介

主要针对国内银行的银行卡识别系统，也支持国外银行。

## 功能特性 ⭐

- [x] 银行卡矫正
- [x] YOLOv8 实现卡号/有效期区域定位
- [x] CRNN 实现端到端卡号和有效期识别
- [x] 根据卡号确定银行名称等信息
- [x] 可操作的互动型界面

## 项目结构

```
Bank-Card-v1/
├── CreditCard-OCR/          # 主要代码目录
│   ├── crnn/               # CRNN模型相关
│   ├── yolo/               # YOLO定位相关
│   ├── datasets/           # 数据集
│   ├── models/             # 模型文件
│   ├── scripts/            # 辅助脚本
│   ├── Utils/              # 工具类
│   └── gui.py              # GUI界面
└── README.md               # 项目说明
```

## 使用说明

详细使用说明请参考 [CreditCard-OCR/README.MD](CreditCard-OCR/README.MD)

## 模型性能

- 卡号识别率: 94%+
- 有效期识别: 数据集较小，泛化能力有限

## 训练参数

- Epoch: 100
- Batch Size: 64
- Learning Rate: 0.0001
- Optimizer: Adam
- 最佳性能: 50-60 epoch

## 数据集

- YOLO定位数据集: [Roboflow](https://universe.roboflow.com/i-need-this-for-graduation/creditcardorbankcard)
- 标注了600+图片，增强3x到1500+张
- CRNN训练集: 真实数据增强 + 模拟数据集，达到5万+张

## 技术栈

- Python
- PyTorch
- YOLOv8
- CRNN
- LCNet (backbone)
- PyQt5 (GUI)

## 开发环境

- Python 3.x
- CUDA 11.4+
- PyTorch
- 更多依赖请参考 requirements.txt

## 许可证

毕业设计项目

## 贡献

欢迎提交 Issue 和 Pull Request

