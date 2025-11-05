import torch
from config import *
from model import CRNN
from torch.nn import CTCLoss
from ctc_decoder import ctc_decode
from torch.utils.data import DataLoader
from dataset import CardDataset, dynamic_pad_collate_fn

def process(crnn, dataloader, criterion, device, decode_method, beam_size, label2char=None, strip_edge_slash=False):
    total_count = 0
    total_loss = 0
    total_correct = 0
    
    all_preds = []
    all_reals = []
    wrong_cases = []  # 存储错误样本
    
    for data in dataloader:
        images, targets, target_lengths = [i.to(device) for i in data]

        logits = crnn(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        
        # 🔥 解码时传入 label2char
        preds = ctc_decode(log_probs, label2char=label2char, 
                          method=decode_method, beam_size=beam_size)
        
        reals = targets.cpu().detach().numpy().tolist()
        target_lengths = target_lengths.cpu().detach().numpy().tolist()

        total_count += batch_size
        total_loss += loss.item()
        
        target_length_counter = 0
        for pred, target_length in zip(preds, target_lengths):
            real = reals[target_length_counter:target_length_counter + target_length]
            target_length_counter += target_length
            
            # 🔥 转换为字符串
            if label2char:
                pred_str = ''.join(pred)  # pred已经是字符列表
                real_str = ''.join([label2char[l] for l in real])
            else:
                pred_str = str(pred)
                real_str = str(real)

            # 测试集：若预测首/尾为'/'则去掉
            if strip_edge_slash and isinstance(pred_str, str):
                if pred_str.startswith('/') or pred_str.endswith('/'):
                    pred_str = pred_str.strip('/')
            
            all_preds.append(pred_str)
            all_reals.append(real_str)
            
            if pred_str == real_str:
                total_correct += 1
            else:
                # 记录错误样本
                wrong_cases.append((real_str, pred_str))
    
    return total_loss / total_count, total_correct / total_count, all_preds, all_reals, wrong_cases


def evaluate(crnn, data_dir, label2char=None, show_samples=10):
    test_dataset = CardDataset(image_dir=data_dir+'/test', mode='test',
                              img_height=img_height, img_width=img_width)
    val_dataset = CardDataset(image_dir=data_dir+'/val', mode='val',
                             img_height=img_height, img_width=img_width)
    test_loader = DataLoader(dataset=test_dataset, batch_size=eval_batch_size,
                            shuffle=True, num_workers=num_workers,
                            collate_fn=dynamic_pad_collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=eval_batch_size,
                           shuffle=True, num_workers=num_workers,
                           collate_fn=dynamic_pad_collate_fn)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)
    crnn.eval()

    with torch.no_grad():
        # 验证集
        val_loss, val_accu, val_preds, val_reals, val_wrong = process(
            crnn, val_loader, criterion, device, decode_method, beam_size, label2char,
            strip_edge_slash=False
        )
        
        # 测试集
        test_loss, test_accu, test_preds, test_reals, test_wrong = process(
            crnn, test_loader, criterion, device, decode_method, beam_size, label2char,
            strip_edge_slash=True
        )
        
        # 🔥 打印验证集样本（前show_samples个）
        print(f"\n{'='*80}")
        print("验证集样本展示:")
        print(f"{'='*80}")
        for i in range(min(show_samples, len(val_preds))):
            match = "✅" if val_preds[i] == val_reals[i] else "❌"
            print(f"[{i:02d}] {match} 真实: '{val_reals[i]}' | 预测: '{val_preds[i]}'")
        
        # 🔥 打印验证集错误样本统计
        if val_wrong:
            print(f"\n❌ 验证集错误样本数: {len(val_wrong)} / {len(val_preds)}")
            print("前10个错误样本:")
            for i, (real, pred) in enumerate(val_wrong[:10]):
                print(f"    真实: '{real}' | 预测: '{pred}'")
        
        # 🔥 打印测试集样本（全部样本）
        print(f"\n{'='*80}")
        print("测试集样本展示:")
        print(f"{'='*80}")
        for i in range(len(test_preds)):
            match = "✅" if test_preds[i] == test_reals[i] else "❌"
            print(f"[{i:02d}] {match} 真实: '{test_reals[i]}' | 预测: '{test_preds[i]}'")
        
        # 🔥 打印测试集错误样本统计
        if test_wrong:
            print(f"\n❌ 测试集错误样本数: {len(test_wrong)} / {len(test_preds)}")
            print("前10个错误样本:")
            for i, (real, pred) in enumerate(test_wrong[:10]):
                print(f"    真实: '{real}' | 预测: '{pred}'")
    
    return val_loss, val_accu, test_loss, test_accu


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'D:\Softwares\Python\CreditCard-OCR\datasets/recognition\processed'
    
    # 🔥 获取 label2char
    temp_dataset = CardDataset(image_dir=data_dir+'/test', mode='val',
                              img_height=img_height, img_width=img_width)
    label2char = temp_dataset.LABEL2CHAR
    
    crnn = CRNN(1, 32, 512, 11)
    crnn.load_state_dict(torch.load('./runs/recognition/run3/checkpoints/crnn best.pt', 
                                    map_location=device))
    crnn.to(device)
    
    test_loss, accuracy, val_loss, val_accu = evaluate(crnn, data_dir, 
                                                        label2char=label2char, 
                                                        show_samples=10)
    
    print(f"\n{'='*80}")
    print("最终结果:")
    print(f"{'='*80}")
    print(f'val_loss: {val_loss:.4f}')
    print(f'val_accu: {val_accu:.2%}')
    print(f'test_loss: {test_loss:.4f}')
    print(f'test_accu: {accuracy:.2%}')
