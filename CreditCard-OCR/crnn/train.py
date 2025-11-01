import os
import csv
import torch
from config import *
from model import CRNN
import torch.optim as optim
from torch.nn import CTCLoss
from evaluate import evaluate
from torch.utils.data import DataLoader
from dataset import CardDataset, cardnumber_collate_fn

def train_batch(crnn, data, optimizer, criterion, device):
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
    print(f'device: {device}')
    
    # 加载数据集
    train_dataset = CardDataset(image_dir=data_dir+'/train', mode='train',
                                    img_height=img_height, img_width=img_width)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=cardnumber_collate_fn)
    
    num_class = len(CardDataset.LABEL2CHAR) + 1
    print(f"\nCharacter Set: '{CardDataset.CHARS}'")
    print(f"Number of Classes: {num_class} (including blank)\n")
    
    crnn = CRNN(1, img_height, img_width, num_class,
                map_to_seq_hidden=map_to_seq_hidden,
                rnn_hidden=rnn_hidden,
                leaky_relu=leaky_relu,
                backbone=backbone)
    print(crnn)

    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
        print(f"✅ Loaded checkpoint: {reload_checkpoint}")
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
    
    # 保存路径
    os.makedirs('./runs/recognition', exist_ok=True)
    run = 1
    while os.path.exists('./runs/recognition/run'+str(run)):
        run += 1
    os.makedirs('./runs/recognition/run'+str(run)+'/checkpoints', exist_ok=True)
    os.makedirs('./runs/recognition/run'+str(run)+'/visualizations', exist_ok=True)  # ⭐ 创建可视化目录
    save_path = './runs/recognition/run'+str(run)
    
    print(f"Save Path: {save_path}\n")

    # 训练
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
            print('train_batch_loss[', index, ' / ', length, ']: ', loss / train_size, end="\r")
            index += 1
        
        print('total_train_loss: ', total_train_loss / total_train_count)
        temp = []
        temp.append(epoch)
        temp.append(total_train_loss / total_train_count)

        torch.save(crnn.state_dict(), save_path + '/checkpoints/crnn last.pt')
        
        # ⭐ 每轮都启用调试和可视化
        vis_dir = save_path + f'/visualizations/epoch_{epoch}'
        os.makedirs(vis_dir, exist_ok=True)
        
        test_loss, test_accuracy, val_loss, val_accuracy = evaluate(
            crnn, data_dir, 
            debug=True,    # ⭐ 所有epoch都启用调试
            save_dir=vis_dir  # ⭐ 每轮保存可视化结果
        )
        
        temp.append(val_loss)
        temp.append(val_accuracy)
        temp.append(test_loss)
        temp.append(test_accuracy)
        data.append(temp)
        
        print(f'\nval_loss: {val_loss:.4f}')
        print(f'val_accu: {val_accuracy:.4f}')
        print(f'test_loss: {test_loss:.4f}')
        print(f'accuracy: {test_accuracy:.4f}')

        with open(save_path + '/results.csv', 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch','train_loss','val_loss', 'val_accu', 'test_loss', 'accuracy'])
            writer.writerows(data)
        
        # 保存最好的模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            torch.save(crnn.state_dict(), save_path + '/checkpoints/crnn best.pt')
            print('save model at ' + save_path + '/checkpoints/crnn best.pt')
        # earlystop策略
        elif best_epoch is not None and epoch - best_epoch > early_stop:
            print('early stopped because not improved for {} epochs'.format(early_stop))
            break

    print('\n' + "="*60)
    print('Training Completed')
    print("="*60)
    print('best epoch:', best_epoch)
    print('best accuracy:', best_accuracy)


if __name__ == '__main__':
    main()
