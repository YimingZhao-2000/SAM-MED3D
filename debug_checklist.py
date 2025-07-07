#!/usr/bin/env python3
"""
快速自检 Checklist
检查 MedSAM2 推理环境的所有潜在问题
"""

import os
import sys
from pathlib import Path

def print_header(title):
    print(f"\n{'='*50}")
    print(f"🔍 {title}")
    print(f"{'='*50}")

def check_1_cwd():
    """① 检查当前工作目录"""
    print_header("① 检查当前工作目录")
    
    cwd = os.getcwd()
    print(f"当前工作目录: {cwd}")
    
    # 检查是否在项目根目录
    if Path("sam2").exists() and Path("checkpoints").exists():
        print("✅ 在项目根目录")
    else:
        print("❌ 不在项目根目录")
        print("建议: cd 到项目根目录再运行")
    
    return cwd

def check_2_config_files():
    """② 检查配置文件是否存在"""
    print_header("② 检查配置文件")
    
    config_paths = [
        "sam2/configs/sam2.1_hiera_t512.yaml",
        "sam2/configs/sam2.1_hiera_tiny_finetune512.yaml",
        "configs/sam2.1/sam2.1_hiera_t512.yaml"
    ]
    
    for config_path in config_paths:
        path = Path(config_path)
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"{status} {config_path}")
        if exists:
            print(f"   绝对路径: {path.resolve()}")
    
    # 检查sam2/configs目录
    configs_dir = Path("sam2/configs")
    if configs_dir.exists():
        print(f"\n📁 {configs_dir} 目录内容:")
        for file in configs_dir.glob("*.yaml"):
            print(f"   - {file.name}")
    else:
        print(f"❌ {configs_dir} 目录不存在")

def check_3_package_installation():
    """③ 检查包安装状态"""
    print_header("③ 检查包安装状态")
    
    # 检查sam2包
    try:
        import sam2
        print(f"✅ sam2 包已安装: {sam2.__file__}")
    except ImportError:
        print("❌ sam2 包未安装")
        return False
    
    # 检查是否editable安装
    try:
        import importlib.util
        spec = importlib.util.find_spec("sam2")
        if spec and spec.origin:
            origin = Path(spec.origin)
            if "site-packages" in str(origin):
                print("⚠️  sam2 是普通安装 (非editable)")
                print("建议: pip install -e .")
            else:
                print("✅ sam2 是editable安装")
        else:
            print("❓ 无法确定sam2安装方式")
    except Exception as e:
        print(f"⚠️  检查安装方式时出错: {e}")
    
    # 检查关键模块
    modules_to_check = [
        "sam2.build_sam",
        "sam2.sam2_image_predictor", 
        "sam2.configs"
    ]
    
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"✅ {module} 可导入")
        except ImportError as e:
            print(f"❌ {module} 导入失败: {e}")
    
    return True

def check_4_hydra_version():
    """④ 检查Hydra版本"""
    print_header("④ 检查Hydra版本")
    
    try:
        import hydra
        print(f"✅ Hydra版本: {hydra.__version__}")
        
        # 检查Hydra初始化
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            print("✅ Hydra已初始化")
        else:
            print("⚠️  Hydra未初始化")
            
    except ImportError:
        print("❌ Hydra未安装")
        print("建议: pip install hydra-core")
    except Exception as e:
        print(f"⚠️  检查Hydra时出错: {e}")

def check_5_config_loading():
    """⑤ 测试配置文件加载"""
    print_header("⑤ 测试配置文件加载")
    
    config_paths = [
        "sam2/configs/sam2.1_hiera_t512.yaml",
        "sam2/configs/sam2.1_hiera_tiny_finetune512.yaml"
    ]
    
    for config_path in config_paths:
        path = Path(config_path)
        if path.exists():
            print(f"\n🔍 测试加载: {config_path}")
            print(f"   绝对路径: {path.resolve()}")
            
            try:
                # 测试OmegaConf加载
                from omegaconf import OmegaConf
                cfg = OmegaConf.load(path)
                print("   ✅ OmegaConf加载成功")
                
                # 检查关键字段
                if 'model' in cfg:
                    print("   ✅ 包含model配置")
                else:
                    print("   ⚠️  缺少model配置")
                    
            except Exception as e:
                print(f"   ❌ 加载失败: {e}")

def check_6_build_sam_test():
    """⑥ 测试build_sam函数"""
    print_header("⑥ 测试build_sam函数")
    
    try:
        from sam2.build_sam import build_sam2
        print("✅ build_sam2 函数可导入")
        
        # 测试配置文件路径
        config_file = "sam2/configs/sam2.1_hiera_t512.yaml"
        if Path(config_file).exists():
            print(f"✅ 配置文件存在: {config_file}")
        else:
            print(f"❌ 配置文件不存在: {config_file}")
            
    except ImportError as e:
        print(f"❌ build_sam2 导入失败: {e}")

def main():
    """主检查函数"""
    print("🚀 MedSAM2 环境自检清单")
    print("="*50)
    
    # 执行所有检查
    cwd = check_1_cwd()
    check_2_config_files()
    package_ok = check_3_package_installation()
    check_4_hydra_version()
    check_5_config_loading()
    check_6_build_sam_test()
    
    # 总结
    print_header("📊 检查总结")
    print("如果看到 ❌ 或 ⚠️，请按以下顺序修复:")
    print("1. 确保在项目根目录运行")
    print("2. 执行: pip install -e .")
    print("3. 检查配置文件路径")
    print("4. 重新运行推理脚本")

if __name__ == "__main__":
    main() 