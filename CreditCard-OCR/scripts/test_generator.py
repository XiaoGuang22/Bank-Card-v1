"""
测试数据生成器是否正常工作
"""
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Utils.random_dataset import HybridBackgroundGeneratorV5

def test_generator():
    """测试生成器"""
    print("🧪 测试数据生成器...\n")
    
    # 初始化生成器（需要调整路径）
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    font_path = os.path.join(BASE_DIR, 'Font/Farrington-7B.ttf')
    real_bg_dir = os.path.join(BASE_DIR, 'data/true_background')
    
    # 检查必要文件
    if not os.path.exists(font_path):
        print(f"❌ 字体文件不存在: {font_path}")
        print("请确保字体文件存在于 Font/ 目录")
        return False
    
    if not os.path.exists(real_bg_dir):
        print(f"❌ 背景图片目录不存在: {real_bg_dir}")
        print("请确保真实背景图片存在于 data/true_background/ 目录")
        return False
    
    try:
        generator = HybridBackgroundGeneratorV5(font_path, real_bg_dir)
        
        # 测试生成一个样本
        print("生成测试样本...")
        test_output = os.path.join(BASE_DIR, 'datasets', 'test_output')
        os.makedirs(test_output, exist_ok=True)
        
        img_array, formatted_number = generator.generate_sample(
            save_path=os.path.join(test_output, 'test_sample.png'),
            sample_index=1
        )
        
        print(f"\n✅ 测试成功！")
        print(f"生成的卡号格式: {formatted_number}")
        print(f"文件保存位置: {test_output}/test_sample.png")
        
        # 检查格式
        if '/' in formatted_number:
            print("✅ 格式正确：包含 '/' 分隔符")
        else:
            print("❌ 格式错误：缺少 '/' 分隔符")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_generator()
    sys.exit(0 if success else 1)

