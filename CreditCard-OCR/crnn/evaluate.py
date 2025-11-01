import torch
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示窗口

from config import *
from model import CRNN
from torch.nn import CTCLoss
from ctc_decoder import ctc_decode
from torch.utils.data import DataLoader
from dataset import CardDataset, cardnumber_collate_fn


def visualize_predictions(images, targets, target_lengths, preds, save_path, num_samples=4):
    """
    可视化预测结果
    
    Args:
        images: (batch, 1, H, W) 图像张量
        targets: 展平的目标标签
        target_lengths: 每个样本的标签长度
        preds: 预测结果列表
        save_path: 保存路径
        num_samples: 可视化的样本数量
    """
    batch_size = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(batch_size, 1, figsize=(12, 3 * batch_size))
    if batch_size == 1:
        axes = [axes]
    
    target_length_counter = 0
    for i in range(batch_size):
        # 获取图像
        img = images[i, 0].cpu().numpy()  # (H, W)
        
        # 获取真实标签
        target_length = target_lengths[i].item()
        real = targets[target_length_counter:target_length_counter + target_length].cpu().numpy().tolist()
        target_length_counter += target_length
        
        # 解码真实标签和预测标签
        real_text = ''.join([CardDataset.LABEL2CHAR.get(label, '?') for label in real])
        pred_text = ''.join([CardDataset.LABEL2CHAR.get(label, '?') for label in preds[i]])
        
        # 判断是否正确
        is_correct = (preds[i] == real)
        color = 'green' if is_correct else 'red'
        status = '[OK]' if is_correct else '[ERR]'  # 使用文本替代emoji避免字体问题
        
        # 显示图像
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        # 使用英文标签避免字体问题
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


def process(crnn, dataloader, criterion, device, decode_method, beam_size, debug=False, save_dir=None, mode='val'):
    """
    Args:
        debug: 是否打印调试信息
        save_dir: 保存可视化结果的目录
        mode: 'val' 或 'test'
    """
    total_count = 0
    total_loss = 0
    total_correct = 0
    wrong_cases = []
    
    # ✅ 添加：用于可视化的第一个batch
    first_batch_data = None
    
    for batch_idx, data in enumerate(dataloader):
        images, targets, target_lengths, original_widths = [i.to(device) for i in data]

        logits = crnn(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.clamp((original_widths // 4) - 1, min=1, max=logits.size(0)).to(device)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
        reals = targets.cpu().detach().numpy().tolist()
        target_lengths_list = target_lengths.cpu().detach().numpy().tolist()

        total_count += batch_size
        total_loss += loss.item()
        target_length_counter = 0
        
        for i, (pred, target_length) in enumerate(zip(preds, target_lengths_list)):
            real = reals[target_length_counter:target_length_counter + target_length]
            target_length_counter += target_length
            
            if pred == real:
                total_correct += 1
            else:
                wrong_cases.append((real, pred))
            
            # ✅ 打印调试信息（前3个batch）
            if debug and batch_idx < 3:
                real_text = ''.join([CardDataset.LABEL2CHAR.get(label, '?') for label in real])
                pred_text = ''.join([CardDataset.LABEL2CHAR.get(label, '?') for label in pred])
                status = "✅" if pred == real else "❌"
                print(f"  [{batch_idx}-{i}] {status} 真实: '{real_text}' | 预测: '{pred_text}'")
        
        # ✅ 保存第一个batch用于可视化
        if batch_idx == 0:
            first_batch_data = (images, targets, target_lengths, preds)
    
    # ✅ 可视化第一个batch
    if save_dir and first_batch_data:
        images, targets, target_lengths, preds = first_batch_data
        visualize_predictions(
            images, targets, target_lengths, preds,
            save_path=f"{save_dir}/{mode}_predictions.png",
            num_samples=4
        )
    
    # ✅ 打印错误样本统计
    if debug and len(wrong_cases) > 0:
        print(f"\n错误样本数: {len(wrong_cases)} / {total_count}")
        print("前10个错误样本:")
        for real, pred in wrong_cases[:10]:
            real_text = ''.join([CardDataset.LABEL2CHAR.get(label, '?') for label in real])
            pred_text = ''.join([CardDataset.LABEL2CHAR.get(label, '?') for label in pred])
            print(f"  真实: '{real_text}' | 预测: '{pred_text}'")
    
    return total_loss / total_count, total_correct / total_count


def evaluate(crnn, data_dir, debug=False, save_dir=None):
    """
    Args:
        debug: 是否打印调试信息
        save_dir: 保存可视化结果的目录
    """
    test_dataset = CardDataset(image_dir=data_dir+'/test', mode='test', img_height=img_height, img_width=img_width)
    val_dataset = CardDataset(image_dir=data_dir+'/val', mode='val', img_height=img_height, img_width=img_width)
    test_loader = DataLoader(dataset=test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, collate_fn=cardnumber_collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, collate_fn=cardnumber_collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)
    crnn.eval()

    with torch.no_grad():
        if debug:
            print("\n" + "="*60)
            print("验证集预测结果（前3个batch）")
            print("="*60)
        val_loss, val_accuracy = process(crnn, val_loader, criterion, device, decode_method, beam_size, debug=debug, save_dir=save_dir, mode='val')
        
        if debug:
            print("\n" + "="*60)
            print("测试集预测结果（前3个batch）")
            print("="*60)
        test_loss, test_accuracy = process(crnn, test_loader, criterion, device, decode_method, beam_size, debug=debug, save_dir=save_dir, mode='test')
        
    return test_loss, test_accuracy, val_loss, val_accuracy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'D:\\Softwares\\Python\\CreditCard-OCR\\datasets\\recognition\\processed'
    crnn = CRNN(1, 32, 512, 11)
    crnn.load_state_dict(torch.load('./runs/recognition/run3/checkpoints/crnn best.pt', map_location=device))
    crnn.to(device)
    
    # ✅ 启用调试模式和可视化
    test_loss, test_accuracy, val_loss, val_accuracy = evaluate(crnn, data_dir, debug=True, save_dir='./visualization')
    
    print('\ntest_loss: ', test_loss)
    print('test_accuracy: ', test_accuracy)
    print('val_loss: ', val_loss)
    print('val_accu: ', val_accuracy)
