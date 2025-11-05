import os
import csv
import torch
from config import *
from model import CRNN
import torch.optim as optim
from torch.nn import CTCLoss
from evaluate import evaluate
from torch.utils.data import DataLoader
from dataset import CardDataset, dynamic_pad_collate_fn

def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    
    # 加载数据集和模型
    train_dataset = CardDataset(image_dir=data_dir+'/train', mode='train',
                                img_height=img_height, img_width=img_width)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dynamic_pad_collate_fn)
    
    # 🔥 获取 label2char
    label2char = CardDataset.LABEL2CHAR
    print(f"字符集: {CardDataset.CHARS}")
    print(f"类别数: {len(label2char) + 1} (包含blank)")
    
    num_class = len(label2char) + 1
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=map_to_seq_hidden,
                rnn_hidden=rnn_hidden,
                leaky_relu=leaky_relu,
                backbone=backbone)
    print(crnn)

    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)
    
    # 优化器和损失函数
    if optim_config == 'adam':
        optimizer = optim.Adam(crnn.parameters(), lr=lr)
    elif optim_config == 'sgd':
        optimizer = optim.SGD(crnn.parameters(), lr=lr)
    elif optim_config == 'rmsprop':
        optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    best_accuracy = -1
    best_epoch = None
    data = []
    
    # 保存路径（相对路径，健壮创建）
    recog_root = os.path.join('.', 'runs', 'recognition')
    os.makedirs(recog_root, exist_ok=True)
    run = 1
    while os.path.exists(os.path.join(recog_root, 'run'+str(run))):
        run += 1
    save_path = os.path.join(recog_root, 'run'+str(run))
    os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)

    # 训练
    for epoch in range(1, epochs + 1):
        print(f'\n{"="*80}')
        print(f'Epoch: {epoch}/{epochs}')
        print(f'{"="*80}')
        
        total_train_loss = 0.
        total_train_count = 0
        index = 1
        length = len(train_loader)
        
        # 一个epoch的训练
        for train_data in train_loader: 
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)
            total_train_loss += loss
            total_train_count += train_size
            
            # 🔥 只保留一行进度显示，最后一个batch会保留在屏幕上
            print(f'train_batch_loss[{index:3d} / {length:3d}]: {loss / train_size:.4f}', 
                  end="\r")
            index += 1
        
        # 🔥 最后一个batch的损失会保留，不再单独打印total_train_loss
        print()  # 换行，让最后一行train_batch_loss保留
        
        temp = [epoch, total_train_loss / total_train_count]

        torch.save(crnn.state_dict(), save_path + '/checkpoints/crnn last.pt')
        
        # 🔥 评估该epoch的结果（传入 label2char，显示10个样本）
        val_loss, val_accu, test_loss, test_accu = evaluate(
            crnn, data_dir, label2char=label2char, show_samples=10
        )
        
        temp.extend([val_loss, val_accu, test_loss, test_accu])
        data.append(temp)
        
        # 🔥 打印总结
        print(f'\n{"="*80}')
        print(f'Epoch {epoch} 总结:')
        print(f'{"="*80}')
        print(f'val_loss: {val_loss:.4f}')
        print(f'val_accu: {val_accu:.2%}')
        print(f'test_loss: {test_loss:.4f}')
        print(f'test_accu: {test_accu:.2%}')

        # 保存结果
        with open(save_path + '/results.csv', 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch','train_loss','val_loss', 'val_accu', 
                           'test_loss', 'test_accu'])
            writer.writerows(data)
        
        # 保存最好的模型
        if test_accu > best_accuracy:
            best_accuracy = test_accu
            best_epoch = epoch
            torch.save(crnn.state_dict(), save_path + '/checkpoints/crnn best.pt')
            print(f'✅ 保存最佳模型 (准确率: {best_accuracy:.2%})')
        
        # earlystop策略
        elif epoch - best_epoch > early_stop:
            print(f'⚠️  Early stopping: {early_stop} epochs without improvement')
            break

    print(f'\n{"="*80}')
    print(f'训练完成!')
    print(f'{"="*80}')
    print(f'最佳 Epoch: {best_epoch}')
    print(f'最佳准确率: {best_accuracy:.2%}')


if __name__ == '__main__':
    main()
