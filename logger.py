"""日志系统 — 提供统一的日志记录功能。

支持按模块名获取 logger，输出到控制台和文件，自动日志轮转。
"""

import logging
import os
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
_DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_root_initialized = False


def _ensure_log_dir():
    if not os.path.exists(_LOG_DIR):
        os.makedirs(_LOG_DIR, exist_ok=True)


def _init_root_logger():
    global _root_initialized
    if _root_initialized:
        return
    _ensure_log_dir()
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    file_handler = RotatingFileHandler(
        os.path.join(_LOG_DIR, "app.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATE_FORMAT))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DEFAULT_DATE_FORMAT))

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    _root_initialized = True


def get_logger(name: str | None = None) -> logging.Logger:
    """获取指定名称的 logger。

    Args:
        name: logger 名称，通常使用 __name__。为 None 时返回根 logger。
    """
    _init_root_logger()
    return logging.getLogger(name)
