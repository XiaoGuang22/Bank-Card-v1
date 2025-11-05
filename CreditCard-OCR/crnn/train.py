import os
import csv
import torch
from config import *
from model import CRNN
import torch.optim as optim
from torch.nn import CTCLoss
from evaluate import evaluate
from torch.utils.data import DataLoader, WeightedRandomSampler  # ⭐ 添加 WeightedRandomSampler
from dataset import CardDataset, cardnumber_collate_fn


# ⭐⭐⭐ 新增：创建加权采样器 ⭐⭐⭐
def create_weighted_sampler(dataset):
    """
    创建加权采样器，为困难样本分配更高的权重
    
    Args:
        dataset: CardDataset实例
    
    Returns:
        WeightedRandomSampler
    """
    weights = []
    
    for is_hard in dataset.is_hard_sample:
        if is_hard:
            # ⭐ 困难样本权重为3（被采样概率是普通样本的3倍）
            weights.append(3.0)
        else:
            # 普通样本权重为1
            weights.append(1.0)
    
    # replacement=True 表示有放回采样（同一样本可能在一个epoch中出现多次）
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    
    return sampler


def train_batch(crnn, data, optimizer, criterion, device):
    """
    训练一个batch
    """
    crnn.train()
    images, targets, target_lengths, original_widths = [d.to(device) for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.clamp((original_widths // 4) - 1, min=1, max=logits.size(0)).to(device)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), max_norm=5.0)  # 梯度裁剪
    optimizer.step()
    return loss.item()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # ⭐ 加载数据集
    train_dataset = CardDataset(
        image_dir=data_dir+'/train', 
        mode='train',
        img_height=img_height, 
        img_width=img_width
    )
    
    # ⭐⭐⭐ 新增：创建加权采样器 ⭐⭐⭐
    weighted_sampler = create_weighted_sampler(train_dataset)
    
    # ⭐⭐⭐ 修改：使用加权采样器（注意：使用sampler时不能设置shuffle=True）⭐⭐⭐
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=weighted_sampler,  # ⭐ 使用加权采样，替代 shuffle=True
        num_workers=num_workers,
        collate_fn=cardnumber_collate_fn
    )
    
    # ⭐ 打印字符集信息
    num_class = len(CardDataset.CHARS) + 1  # 字符数 + blank
    print(f"\n{'='*60}")
    print("数据集信息")
    print(f"{'='*60}")
    print(f"字符集: '{CardDataset.CHARS}'")
    print(f"类别数: {num_class} (包含blank)")
    print(f"BOS标记: '{CardDataset.BOS_CHAR}' (label={CardDataset.BOS_LABEL})")
    print(f"EOS标记: '{CardDataset.EOS_CHAR}' (label={CardDataset.EOS_LABEL})")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"⭐ 困难样本权重: 3.0x（被采样概率是普通样本的3倍）")  # ⭐ 新增提示
    print(f"{'='*60}\n")
    
    # ⭐ 创建模型
    crnn = CRNN(
        1, img_height, img_width, num_class,
        map_to_seq_hidden=map_to_seq_hidden,
        rnn_hidden=rnn_hidden,
        leaky_relu=leaky_relu,
        backbone=backbone
    )
    print(crnn)

    # ⭐ 加载checkpoint（如果有）
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
        print(f"\n✅ 加载checkpoint: {reload_checkpoint}")
    crnn.to(device)
    
    # ⭐ 优化器和损失函数
    if optim_config == 'adam':
        optimizer = optim.Adam(crnn.parameters(), lr=lr)
    elif optim_config == 'sgd':
        optimizer = optim.SGD(crnn.parameters(), lr=lr)
    elif optim_config == 'rmsprop':
        optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    else:
        raise ValueError(f"❌ 不支持的优化器: {optim_config}")
    
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    best_accuracy = -1
    best_epoch = None
    data = []
    
    # ⭐ 创建保存目录
    os.makedirs('./runs/recognition', exist_ok=True)
    run = 1
    while os.path.exists('./runs/recognition/run'+str(run)):
        run += 1
    os.makedirs('./runs/recognition/run'+str(run)+'/checkpoints', exist_ok=True)
    os.makedirs('./runs/recognition/run'+str(run)+'/visualizations', exist_ok=True)
    save_path = './runs/recognition/run'+str(run)
    
    print(f"\n保存路径: {save_path}\n")

    # ⭐ 开始训练
    print(f"{'='*60}")
    print("开始训练")
    print(f"{'='*60}\n")
    
    for epoch in range(1, epochs + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{epochs}')
        print("="*60)
        
        total_train_loss = 0.
        total_train_count = 0
        index = 1
        length = len(train_loader)
        
        # 训练一个epoch
        for train_data in train_loader: 
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)
            total_train_loss += loss
            total_train_count += train_size
            print(f'train_batch_loss [{index}/{length}]: {loss / train_size:.4f}', end="\r")
            index += 1
        
        avg_train_loss = total_train_loss / total_train_count
        print(f'\ntotal_train_loss: {avg_train_loss:.4f}')
        
        temp = []
        temp.append(epoch)
        temp.append(avg_train_loss)

        # 保存最新的模型
        torch.save(crnn.state_dict(), save_path + '/checkpoints/crnn last.pt')
        
        # ⭐ 每轮都启用调试和可视化
        vis_dir = save_path + f'/visualizations/epoch_{epoch}'
        os.makedirs(vis_dir, exist_ok=True)
        
        # ⭐⭐⭐ 修复：使用正确的BOS和EOS标记 ⭐⭐⭐
        test_loss, test_accuracy, val_loss, val_accuracy = evaluate(
            crnn, data_dir, 
            debug=True,
            save_dir=vis_dir,
            bos_label=CardDataset.BOS_LABEL,  # ✅ 使用 '<' 的label
            eos_label=CardDataset.EOS_LABEL   # ✅ 使用 '>' 的label
        )
        
        temp.append(val_loss)
        temp.append(val_accuracy)
        temp.append(test_loss)
        temp.append(test_accuracy)
        data.append(temp)
        
        print(f'\nval_loss: {val_loss:.4f}')
        print(f'val_accu: {val_accuracy*100:.2f}%')
        print(f'test_loss: {test_loss:.4f}')
        print(f'test_accu: {test_accuracy*100:.2f}%')

        # 保存结果到CSV
        with open(save_path + '/results.csv', 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch','train_loss','val_loss', 'val_accu', 'test_loss', 'test_accu'])
            writer.writerows(data)
        
        # 保存最好的模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            torch.save(crnn.state_dict(), save_path + '/checkpoints/crnn best.pt')
            print(f'✅ 保存最佳模型到: {save_path}/checkpoints/crnn best.pt')
        
        # ⭐ Early stopping策略
        elif best_epoch is not None and epoch - best_epoch > early_stop:
            print(f'\n⚠️ Early stopping: 已经{early_stop}个epoch没有提升')
            break

    # ⭐ 训练完成
    print('\n' + "="*60)
    print('训练完成')
    print("="*60)
    print(f'最佳epoch: {best_epoch}')
    print(f'最佳准确率: {best_accuracy*100:.2f}%')
    print(f'结果保存在: {save_path}')
    print("="*60)


if __name__ == '__main__':
    main()
