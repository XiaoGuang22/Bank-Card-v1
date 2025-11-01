"""
给所有标签文件添加下划线结束符
在所有CardNumberlabel列的末尾添加 '_'
"""
import pandas as pd
import os


def add_underscore_to_labels(file_path):
    """
    读取Excel文件，给标签列添加下划线
    
    Args:
        file_path: Excel文件路径
    """
    print(f"📝 处理文件: {file_path}")
    
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    print(f"   原始数据:")
    print(f"   - 行数: {len(df)}")
    print(f"   - 列名: {df.columns.tolist()}")
    print(f"   前3个标签: {df['CardNumberlabel'].head(3).tolist()}")
    
    # 统计修改前的标签
    before_count = 0
    
    # 给每一行的CardNumberlabel添加下划线（如果还没有的话）
    for idx in range(len(df)):
        label = str(df.at[idx, 'CardNumberlabel'])
        if not label.endswith('_'):
            df.at[idx, 'CardNumberlabel'] = label + '_'
            before_count += 1
    
    print(f"   修改了 {before_count} 个标签")
    print(f"   修改后前3个标签: {df['CardNumberlabel'].head(3).tolist()}")
    
    # 保存文件（原地修改）
    df.to_excel(file_path, index=False)
    print(f"   ✅ 已保存到: {file_path}\n")


def main():
    """
    主函数：处理所有标签文件
    """
    print("🚀 开始给标签添加下划线结束符...\n")
    
    # 定义所有标签文件路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    label_files = [
        os.path.join(base_dir, 'datasets', 'train', 'train_labels.xlsx'),
        os.path.join(base_dir, 'datasets', 'val', 'val_labels.xlsx'),
        os.path.join(base_dir, 'datasets', 'test', 'test_labels.xlsx'),
    ]
    
    # 检查文件是否存在并处理
    success_count = 0
    for file_path in label_files:
        if os.path.exists(file_path):
            try:
                add_underscore_to_labels(file_path)
                success_count += 1
            except Exception as e:
                print(f"   ❌ 处理失败: {e}\n")
        else:
            print(f"⚠️  文件不存在: {file_path}\n")
    
    print(f"🎉 完成！成功处理 {success_count}/{len(label_files)} 个文件")


if __name__ == '__main__':
    main()

