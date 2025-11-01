"""
数据集划分脚本：将训练集分为train和val
"""
import os
import shutil
import pandas as pd
import random

def split_dataset(source_dir, train_ratio=0.8, seed=42):
    """
    将数据集分为train和val
    
    Args:
        source_dir: 源数据目录（包含images和labels.xlsx）
        train_ratio: 训练集比例，默认0.8（即20%作为验证集）
        seed: 随机种子
    """
    random.seed(seed)
    
    # 读取标签文件
    excel_path = os.path.join(source_dir, 'labels.xlsx')
    if not os.path.exists(excel_path):
        print(f"❌ 未找到标签文件: {excel_path}")
        return
    
    df = pd.read_excel(excel_path, engine='openpyxl')
    print(f"✅ 读取到 {len(df)} 条标签记录")
    
    # 随机打乱
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # 计算划分点
    split_idx = int(len(df_shuffled) * train_ratio)
    
    train_df = df_shuffled[:split_idx].copy()
    val_df = df_shuffled[split_idx:].copy()
    
    print(f"📊 数据划分:")
    print(f"   训练集: {len(train_df)} 条 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   验证集: {len(val_df)} 条 ({len(val_df)/len(df)*100:.1f}%)")
    
    # 创建输出目录
    train_dir = os.path.join(os.path.dirname(source_dir), 'train')
    val_dir = os.path.join(os.path.dirname(source_dir), 'val')
    
    # 清空并重建目录
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 复制图片并保存标签
    def copy_dataset(df, target_dir, dataset_name):
        copied = 0
        skipped = 0
        
        for _, row in df.iterrows():
            filename = row['filename']
            src_img = os.path.join(source_dir, filename)
            dst_img = os.path.join(target_dir, filename)
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
                copied += 1
            else:
                skipped += 1
                print(f"⚠️  跳过：{filename} 不存在")
        
        # 保存标签
        label_file = os.path.join(target_dir, 'labels.xlsx')
        df.to_excel(label_file, index=False, engine='openpyxl')
        
        print(f"✅ {dataset_name}: 复制了 {copied} 张图片到 {target_dir}")
        if skipped > 0:
            print(f"⚠️  {dataset_name}: 跳过了 {skipped} 个不存在的文件")
    
    # 复制训练集和验证集
    copy_dataset(train_df, train_dir, "训练集")
    copy_dataset(val_df, val_dir, "验证集")
    
    print("\n✅ 数据集划分完成！")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        source_dir = sys.argv[1]
        train_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    else:
        # 默认路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        source_dir = os.path.join(os.path.dirname(current_dir), 'datasets')
        train_ratio = 0.8
        print(f"使用默认路径: {source_dir}")
    
    split_dataset(source_dir, train_ratio=train_ratio)

