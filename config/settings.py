# config/settings.py
"""
项目配置文件
"""

from datetime import datetime
import os

# YOLO模型配置
MODEL_PATH = "./model/best.pt"
CONFIDENCE_THRESHOLD = 0.5

# 机器人控制配置
ALIGNMENT_TOLERANCE = 50
MOVE_SPEED = 0.25
ROTATE_SPEED_P = 0.05
DISTANCE_THRESHOLD = 270
DISTANCE_SENSOR_INDEX = 3

# Marker配置
TARGET_MARKER_NAME = "heart"
MARKER_DISTANCE_THRESHOLD = 0.20

# 日志配置
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, f"robomaster_{datetime.now().strftime('%Y%m%d')}.log")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 其他配置...