import torch
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from config import *
from model import CRNN
from torch.nn import CTCLoss
from ctc_decoder import ctc_decode
from torch.utils.data import DataLoader
from dataset import CardDataset, cardnumber_collate_fn


def visualize_predictions(images, targets, target_lengths, preds, save_path, num_samples=4, bos_label=None, eos_label=None):
    """
    可视化预测结果
    
    Args:
        images: 图像batch
        targets: 目标标签
        target_lengths: 目标长度
        preds: 预测结果
        save_path: 保存路径
        num_samples: 显示样本数
        bos_label: BOS标记的label
        eos_label: EOS标记的label
    """
    batch_size = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(batch_size, 1, figsize=(12, 3 * batch_size))
    if batch_size == 1:
        axes = [axes]
    
    target_length_counter = 0
    for i in range(batch_size):
        img = images[i, 0].cpu().numpy()
        
        target_length = target_lengths[i].item()
        real = targets[target_length_counter:target_length_counter + target_length].cpu().numpy().tolist()
        target_length_counter += target_length
        
        # ⭐ 去除BOS和EOS标记后再显示
        real_without_markers = [
            label for label in real 
            if label != bos_label and label != eos_label
        ]
        
        real_text = ''.join([CardDataset.LABEL2CHAR.get(label, '?') for label in real_without_markers])
        pred_text = ''.join([CardDataset.LABEL2CHAR.get(label, '?') for label in preds[i]])
        
        is_correct = (preds[i] == real_without_markers)
        color = 'green' if is_correct else 'red'
        status = '[OK]' if is_correct else '[ERR]'
        
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        title_parts = [
            f"{status} Real: '{real_text}'" if real_text else f"{status} Real: ''",
            f"Pred: '{pred_text}'" if pred_text else "Pred: ''"
        ]
        axes[i].set_title(
            ' | '.join(title_parts),
            fontsize=12,
            color=color
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 可视化结果已保存到: {save_path}")


def process(crnn, dataloader, criterion, device, decode_method, beam_size, debug=False, save_dir=None, mode='val', bos_label=None, eos_label=None):
    """
    处理一个数据集（验证集或测试集）
    
    Args:
        crnn: CRNN模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        decode_method: 解码方法
        beam_size: beam size
        debug: 是否打印调试信息
        save_dir: 保存可视化结果的目录
        mode: 'val' 或 'test'
        bos_label: BOS标记的label
        eos_label: EOS标记的label
    
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    total_count = 0
    total_loss = 0
    total_correct = 0
    wrong_cases = []
    
    first_batch_data = None
    
    for batch_idx, data in enumerate(dataloader):
        images, targets, target_lengths, original_widths = [i.to(device) for i in data]

        logits = crnn(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.clamp((original_widths // 4) - 1, min=1, max=logits.size(0)).to(device)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        
        # ⭐ ctc_decode返回label序列（已经去除了BOS/EOS）
        preds = ctc_decode(
            log_probs, 
            label2char=None,  # 返回label序列
            blank=0,
            method=decode_method, 
            beam_size=beam_size, 
            bos_label=bos_label,
            eos_label=eos_label
        )
        
        reals = targets.cpu().detach().numpy().tolist()
        target_lengths_list = target_lengths.cpu().detach().numpy().tolist()

        total_count += batch_size
        total_loss += loss.item()
        target_length_counter = 0
        
        for i, (pred, target_length) in enumerate(zip(preds, target_lengths_list)):
            real = reals[target_length_counter:target_length_counter + target_length]
            target_length_counter += target_length
            
            # ⭐ 比较时去除BOS和EOS标记
            real_without_markers = [
                l for l in real 
                if l != bos_label and l != eos_label
            ]
            
            if pred == real_without_markers:
                total_correct += 1
            else:
                wrong_cases.append((real_without_markers, pred))
            
            if debug and batch_idx < 3:
                real_text = ''.join([CardDataset.LABEL2CHAR.get(label, '?') for label in real_without_markers])
                pred_text = ''.join([CardDataset.LABEL2CHAR.get(label, '?') for label in pred])
                status = "✅" if pred == real_without_markers else "❌"
                print(f"  [{batch_idx}-{i}] {status} 真实: '{real_text}' | 预测: '{pred_text}'")
        
        if batch_idx == 0:
            first_batch_data = (images, targets, target_lengths, preds)
    
    # 保存可视化结果
    if save_dir and first_batch_data:
        os.makedirs(save_dir, exist_ok=True)
        images, targets, target_lengths, preds = first_batch_data
        visualize_predictions(
            images, targets, target_lengths, preds,
            save_path=f"{save_dir}/{mode}_predictions.png",
            num_samples=4,
            bos_label=bos_label,
            eos_label=eos_label
        )
    
    # 打印错误样本
    if debug and len(wrong_cases) > 0:
        print(f"\n错误样本数: {len(wrong_cases)} / {total_count}")
        print("前10个错误样本:")
        for real, pred in wrong_cases[:10]:
            real_text = ''.join([CardDataset.LABEL2CHAR.get(label, '?') for label in real])
            pred_text = ''.join([CardDataset.LABEL2CHAR.get(label, '?') for label in pred])
            print(f"  真实: '{real_text}' | 预测: '{pred_text}'")
    
    return total_loss / total_count, total_correct / total_count


def evaluate(crnn, data_dir, debug=False, save_dir=None, bos_label=None, eos_label=None):
    """
    评估模型
    
    Args:
        crnn: CRNN模型
        data_dir: 数据目录
        debug: 是否打印调试信息
        save_dir: 保存可视化结果的目录
        bos_label: BOS标记的label
        eos_label: EOS标记的label
    
    Returns:
        test_loss: 测试集损失
        test_accuracy: 测试集准确率
        val_loss: 验证集损失
        val_accuracy: 验证集准确率
    """
    # 创建数据集
    test_dataset = CardDataset(
        image_dir=os.path.join(data_dir, 'test'), 
        mode='test', 
        img_height=img_height, 
        img_width=img_width
    )
    val_dataset = CardDataset(
        image_dir=os.path.join(data_dir, 'val'), 
        mode='val', 
        img_height=img_height, 
        img_width=img_width
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=eval_batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=cardnumber_collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=eval_batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=cardnumber_collate_fn
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)
    crnn.eval()

    with torch.no_grad():
        if debug:
            print("\n" + "="*60)
            print("验证集预测结果（前3个batch）")
            print("="*60)
        val_loss, val_accuracy = process(
            crnn, val_loader, criterion, device, decode_method, beam_size, 
            debug=debug, save_dir=save_dir, mode='val',
            bos_label=bos_label, eos_label=eos_label
        )
        
        if debug:
            print("\n" + "="*60)
            print("测试集预测结果（前3个batch）")
            print("="*60)
        test_loss, test_accuracy = process(
            crnn, test_loader, criterion, device, decode_method, beam_size, 
            debug=debug, save_dir=save_dir, mode='test',
            bos_label=bos_label, eos_label=eos_label
        )
        
    return test_loss, test_accuracy, val_loss, val_accuracy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # ⭐ 数据目录
    data_dir = 'D:\\Softwares\\Python\\CreditCard-OCR\\datasets\\recognition\\processed'
    
    # ⭐ 创建模型（类别数=14：0-9数字 + / + < + > + blank）
    num_classes = len(CardDataset.CHARS) + 1  # 13个字符 + 1个blank = 14
    crnn = CRNN(1, 32, 512, num_classes)
    
    # ⭐ 加载模型权重
    checkpoint_path = './runs/recognition/run3/checkpoints/crnn best.pt'
    if os.path.exists(checkpoint_path):
        crnn.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ 加载模型: {checkpoint_path}")
    else:
        print(f"❌ 错误：模型文件不存在: {checkpoint_path}")
        print("请先训练模型！")
        exit(1)
    
    crnn.to(device)
    
    # ⭐ 打印模型信息
    print(f"\n模型配置:")
    print(f"  - 类别数: {num_classes}")
    print(f"  - 字符集: '{CardDataset.CHARS}'")
    print(f"  - BOS标记: '{CardDataset.BOS_CHAR}' (label={CardDataset.BOS_LABEL})")
    print(f"  - EOS标记: '{CardDataset.EOS_CHAR}' (label={CardDataset.EOS_LABEL})")
    print(f"  - 解码方法: {decode_method}")
    print(f"  - Beam size: {beam_size}")
    
    # ⭐ 评估
    test_loss, test_accuracy, val_loss, val_accuracy = evaluate(
        crnn, data_dir, debug=True, save_dir='./visualization',
        bos_label=CardDataset.BOS_LABEL,
        eos_label=CardDataset.EOS_LABEL
    )
    
    # ⭐ 打印最终结果
    print('\n' + "="*60)
    print("最终评估结果")
    print("="*60)
    print(f'验证集 Loss: {val_loss:.4f}')
    print(f'验证集 准确率: {val_accuracy*100:.2f}%')
    print(f'测试集 Loss: {test_loss:.4f}')
    print(f'测试集 准确率: {test_accuracy*100:.2f}%')
    print("="*60)
