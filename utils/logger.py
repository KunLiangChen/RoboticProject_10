# utils/logger.py
"""
统一日志管理模块
提供日志配置、格式化和分级输出功能
"""

import logging
import os
from datetime import datetime
from config.settings import LOG_LEVEL, LOG_FILE_PATH


class LoggerManager:
    """日志管理器类"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_logger()
            LoggerManager._initialized = True

    def _setup_logger(self):
        """配置日志器"""
        # 创建日志目录
        log_dir = os.path.dirname(LOG_FILE_PATH)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 创建logger
        self.logger = logging.getLogger('robomaster_yolo')
        self.logger.setLevel(getattr(logging, LOG_LEVEL))

        # 避免重复添加handler
        if not self.logger.handlers:
            # 文件处理器
            file_handler = logging.FileHandler(
                LOG_FILE_PATH,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, LOG_LEVEL))

            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, LOG_LEVEL))

            # 创建格式器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # 设置格式器
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # 添加处理器
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        """获取logger实例"""
        return self.logger


# 全局日志函数
def get_logger():
    """获取全局logger实例"""
    return LoggerManager().get_logger()


def log_info(message):
    """记录INFO级别日志"""
    logger = get_logger()
    logger.info(message)


def log_warning(message):
    """记录WARNING级别日志"""
    logger = get_logger()
    logger.warning(message)


def log_error(message):
    """记录ERROR级别日志"""
    logger = get_logger()
    logger.error(message)


def log_debug(message):
    """记录DEBUG级别日志"""
    logger = get_logger()
    logger.debug(message)


def log_critical(message):
    """记录CRITICAL级别日志"""
    logger = get_logger()
    logger.critical(message)


# 在config/settings.py中需要添加以下配置：
"""
LOG_LEVEL = "INFO"  # 可选: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE_PATH = f"./logs/robomaster_{datetime.now().strftime('%Y%m%d')}.log"
"""

# 使用示例：
"""
from utils.logger import get_logger, log_info, log_error

logger = get_logger()
logger.info("系统启动")

# 或者直接使用便捷函数
log_info("开始执行任务")
log_error("发生错误: {}".format(str(e)))
"""