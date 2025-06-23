#!/usr/bin/env python3
"""
日志工具函数
提供日志保存和管理功能
"""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

def setup_file_logger(
    log_file: Union[str, Path],
    logger_name: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    also_console: bool = True
) -> logging.Logger:
    """
    设置文件日志记录器
    
    Args:
        log_file: 日志文件路径
        logger_name: 日志器名称，如果为None则使用根日志器
        level: 日志级别
        format_string: 日志格式字符串
        also_console: 是否同时输出到控制台
    
    Returns:
        配置好的日志器
    """
    
    # 确保日志目录存在
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 默认格式
    if format_string is None:
        format_string = '%(asctime)s - %(levelname)s - %(message)s'
    
    # 创建格式器
    formatter = logging.Formatter(format_string)
    
    # 获取或创建日志器
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    
    # 清除现有的处理器（避免重复）
    logger.handlers.clear()
    
    # 设置日志级别
    logger.setLevel(level)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 如果需要，添加控制台处理器
    if also_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def create_timestamped_log_file(
    base_name: str,
    log_dir: Union[str, Path] = "logs",
    extension: str = ".log"
) -> Path:
    """
    创建带时间戳的日志文件路径
    
    Args:
        base_name: 基础文件名
        log_dir: 日志目录
        extension: 文件扩展名
    
    Returns:
        带时间戳的日志文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{base_name}_{timestamp}{extension}"
    return log_file

def setup_test_logger(test_name: str, log_dir: str = "logs") -> tuple[logging.Logger, Path]:
    """
    为测试设置专用日志器
    
    Args:
        test_name: 测试名称
        log_dir: 日志目录
    
    Returns:
        (日志器, 日志文件路径)
    """
    log_file = create_timestamped_log_file(test_name, log_dir)
    logger = setup_file_logger(
        log_file=log_file,
        logger_name=test_name,
        level=logging.INFO,
        format_string='%(asctime)s - %(levelname)s - %(message)s',
        also_console=True
    )
    
    logger.info(f"📝 日志保存到: {log_file}")
    logger.info(f"🧪 开始测试: {test_name}")
    logger.info("=" * 60)
    
    return logger, log_file

def log_system_info(logger: logging.Logger):
    """记录系统信息"""
    import platform
    import psutil
    
    logger.info("🖥️  系统信息:")
    logger.info(f"  操作系统: {platform.system()} {platform.release()}")
    logger.info(f"  Python版本: {platform.python_version()}")
    logger.info(f"  CPU核心数: {psutil.cpu_count()}")
    logger.info(f"  内存总量: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # GPU信息
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            for i, gpu in enumerate(gpus):
                logger.info(f"  GPU {i}: {gpu.name} ({gpu.memoryTotal}MB)")
        else:
            logger.info("  GPU: 未检测到GPU")
    except ImportError:
        logger.info("  GPU: 无法检测GPU信息 (GPUtil未安装)")
    except Exception as e:
        logger.info(f"  GPU: 检测失败 ({e})")

def log_test_summary(logger: logging.Logger, success: bool, duration: float, details: dict = None):
    """记录测试总结"""
    logger.info("=" * 60)
    logger.info("📊 测试总结:")
    logger.info(f"  状态: {'✅ 成功' if success else '❌ 失败'}")
    logger.info(f"  耗时: {duration:.2f} 秒")
    
    if details:
        for key, value in details.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)

class LogCapture:
    """日志捕获上下文管理器"""

    def __init__(self, log_file: Union[str, Path]):
        self.log_file = Path(log_file)
        self.original_handlers = []

    def __enter__(self):
        # 保存原始处理器
        root_logger = logging.getLogger()
        self.original_handlers = root_logger.handlers.copy()

        # 设置文件日志
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 添加文件处理器
        root_logger.addHandler(file_handler)

        return self.log_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始处理器
        root_logger = logging.getLogger()
        root_logger.handlers = self.original_handlers

class TerminalOutputCapture:
    """捕获terminal输出并写入日志文件"""

    def __init__(self, log_file: Union[str, Path], logger_name: str = "terminal_capture"):
        self.log_file = Path(log_file)
        self.logger_name = logger_name
        self.original_stdout = None
        self.original_stderr = None
        self.logger = None
        self.file_handler = None

    def __enter__(self):
        # 确保日志目录存在
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # 创建专用日志器
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.INFO)

        # 清除现有处理器
        self.logger.handlers.clear()

        # 创建文件处理器
        self.file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

        # 防止传播到根日志器
        self.logger.propagate = False

        # 保存原始输出流
        import sys
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # 创建输出拦截器
        sys.stdout = self._create_interceptor(self.original_stdout, "STDOUT")
        sys.stderr = self._create_interceptor(self.original_stderr, "STDERR")

        self.logger.info("🔍 开始捕获terminal输出")
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            self.logger.info("🛑 停止捕获terminal输出")

        # 恢复原始输出流
        if self.original_stdout:
            import sys
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr

        # 关闭文件处理器
        if self.file_handler:
            self.file_handler.close()

    def _create_interceptor(self, original_stream, stream_name):
        """创建输出拦截器"""
        class StreamInterceptor:
            def __init__(self, original, logger, name):
                self.original = original
                self.logger = logger
                self.name = name
                self.buffer = ""

            def write(self, text):
                # 写入原始流
                self.original.write(text)
                self.original.flush()

                # 累积到缓冲区
                self.buffer += text

                # 处理完整的行
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    if line.strip():
                        self._log_line(line.strip())

            def _log_line(self, line):
                """记录单行到日志"""
                clean_line = self._clean_text(line)
                if clean_line and self._should_log(clean_line):
                    # 根据内容类型添加不同的标记
                    if self._is_training_step(clean_line):
                        self.logger.info(f"🏃 [TRAIN] {clean_line}")
                    elif self._is_progress(clean_line):
                        self.logger.info(f"📊 [PROGRESS] {clean_line}")
                    elif self._is_checkpoint(clean_line):
                        self.logger.info(f"💾 [CHECKPOINT] {clean_line}")
                    elif self._is_model_param(clean_line):
                        self.logger.info(f"🧠 [MODEL] {clean_line}")
                    elif self._is_system_info(clean_line):
                        self.logger.info(f"⚙️ [SYSTEM] {clean_line}")
                    elif self._is_error(clean_line):
                        self.logger.error(f"❌ [{self.name}] {clean_line}")
                    elif self._is_warning(clean_line):
                        self.logger.warning(f"⚠️ [{self.name}] {clean_line}")
                    else:
                        self.logger.info(f"📄 [{self.name}] {clean_line}")

            def _clean_text(self, text):
                """清理文本"""
                import re
                # 移除ANSI颜色代码
                clean = re.sub(r'\x1b\[[0-9;]*m', '', text)
                # 移除多余空格
                clean = ' '.join(clean.split())
                return clean

            def _should_log(self, text):
                """判断是否应该记录"""
                # 总是记录重要的训练信息
                important_patterns = [
                    'Step', 'loss=', 'grad_norm=', 'param_norm=',
                    'Initialized train state:', 'value:', '@float32', '@bfloat16',
                    'Saving checkpoint', 'Finished', 'Error', 'Exception'
                ]

                if any(pattern in text for pattern in important_patterns):
                    return True

                # 过滤掉一些不重要的输出
                skip_patterns = [
                    'thread=MainThread', 'process=0][thread=',
                    'orbax.checkpoint._src', 'tensorstore',
                    'array_metadata', '.tmp-', 'tmp_directory',
                    'barrier_sync_fn', 'multihost.py'
                ]

                # 如果包含跳过模式，则不记录
                if any(pattern in text for pattern in skip_patterns):
                    return False

                # 默认记录其他内容
                return True

            def _is_training_step(self, text):
                """判断是否是训练步骤"""
                return 'Step' in text and ('loss=' in text or 'grad_norm=' in text)

            def _is_progress(self, text):
                """判断是否是进度信息"""
                return '%|' in text and ('it/s' in text or '/s]' in text)

            def _is_checkpoint(self, text):
                """判断是否是检查点信息"""
                keywords = ['Saving checkpoint', 'checkpoint to', 'Finished saving']
                return any(keyword in text for keyword in keywords)

            def _is_error(self, text):
                """判断是否是错误"""
                return any(word in text.lower() for word in ['error', 'exception', 'traceback'])

            def _is_warning(self, text):
                """判断是否是警告"""
                return any(word in text.lower() for word in ['warning', 'warn'])

            def _is_model_param(self, text):
                """判断是否是模型参数信息"""
                param_indicators = [
                    "'].value:", '@float32', '@bfloat16', '@int32',
                    'PaliGemma', 'llm', 'img', 'Transformer',
                    'Initialized train state:'
                ]
                return any(indicator in text for indicator in param_indicators)

            def _is_system_info(self, text):
                """判断是否是系统信息"""
                system_indicators = [
                    'Initialized data loader:', 'Created', 'Restoring checkpoint',
                    'Finished restoring', 'Running on:', 'GPU memory',
                    'bytes_per_sec:', 'GiB/s'
                ]
                return any(indicator in text for indicator in system_indicators)

            def flush(self):
                # 处理缓冲区剩余内容
                if self.buffer.strip():
                    self._log_line(self.buffer.strip())
                    self.buffer = ""
                self.original.flush()

            def isatty(self):
                return self.original.isatty()

        return StreamInterceptor(original_stream, self.logger, stream_name)

# 示例用法
if __name__ == "__main__":
    # 创建测试日志器
    logger, log_file = setup_test_logger("example_test")
    
    # 记录系统信息
    log_system_info(logger)
    
    # 模拟测试
    import time
    start_time = time.time()
    
    logger.info("开始执行测试...")
    time.sleep(1)  # 模拟测试过程
    logger.info("测试步骤1完成")
    time.sleep(1)
    logger.info("测试步骤2完成")
    
    # 记录总结
    duration = time.time() - start_time
    log_test_summary(logger, True, duration, {"步骤数": 2, "数据量": "1000条"})
    
    print(f"\n日志已保存到: {log_file}")
