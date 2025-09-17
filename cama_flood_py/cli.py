"""
命令行接口模块
"""

import argparse
import sys
from pathlib import Path
import logging

from .model.controller import CaMaFloodModel
from .model.config import create_default_config


def main():
    """主命令行入口"""
    parser = argparse.ArgumentParser(
        description='CaMa-Flood Python - 全球洪水模拟模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  cama-flood run config.yaml              # 运行模拟
  cama-flood create-config config.yaml    # 创建默认配置
  cama-flood --help                       # 显示帮助
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 运行模拟命令
    run_parser = subparsers.add_parser('run', help='运行洪水模拟')
    run_parser.add_argument('config', help='配置文件路径')
    run_parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    # 创建配置命令
    config_parser = subparsers.add_parser('create-config', help='创建默认配置文件')
    config_parser.add_argument('output', help='输出配置文件路径')
    
    # 解析参数
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    # 设置日志级别
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.command == 'run':
            return run_simulation(args.config)
        elif args.command == 'create-config':
            return create_config_file(args.output)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    
    return 0


def run_simulation(config_path: str) -> int:
    """运行模拟"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"配置文件不存在: {config_path}", file=sys.stderr)
        return 1
    
    print(f"开始运行CaMa-Flood模拟...")
    print(f"配置文件: {config_path}")
    
    try:
        with CaMaFloodModel(config_path) as model:
            model.initialize()
            model.run()
            
        print("模拟完成!")
        return 0
        
    except Exception as e:
        print(f"模拟失败: {e}", file=sys.stderr)
        return 1


def create_config_file(output_path: str) -> int:
    """创建配置文件"""
    try:
        create_default_config(output_path)
        print(f"默认配置文件已创建: {output_path}")
        return 0
    except Exception as e:
        print(f"配置文件创建失败: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
