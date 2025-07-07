#!/usr/bin/env python3
"""
快速修复脚本
解决最常见的MedSAM2环境问题
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n🔧 {description}")
    print(f"命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 成功")
            if result.stdout.strip():
                print(f"输出: {result.stdout.strip()}")
        else:
            print(f"❌ 失败: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

def main():
    print("🚀 MedSAM2 快速修复")
    print("="*40)
    
    # 检查当前目录
    cwd = os.getcwd()
    print(f"当前目录: {cwd}")
    
    # 检查是否在项目根目录
    if not (Path("sam2").exists() and Path("checkpoints").exists()):
        print("❌ 不在项目根目录")
        print("请先 cd 到项目根目录")
        return
    
    print("✅ 在项目根目录")
    
    # 修复步骤
    fixes = [
        ("pip install -e .", "安装editable包"),
        ("pip install hydra-core", "安装Hydra"),
        ("pip install omegaconf", "安装OmegaConf"),
        ("pip install pynrrd", "安装NRRD支持"),
    ]
    
    success_count = 0
    for cmd, desc in fixes:
        if run_command(cmd, desc):
            success_count += 1
    
    print(f"\n📊 修复完成: {success_count}/{len(fixes)} 成功")
    
    if success_count == len(fixes):
        print("🎉 所有修复完成！现在可以运行推理了")
        print("\n运行推理:")
        print("python infer_medsam2_ultrasound.py --help")
    else:
        print("⚠️  部分修复失败，请手动检查")

if __name__ == "__main__":
    main() 