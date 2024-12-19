import logging
import colorlog
from logging.handlers import RotatingFileHandler

def setup_logger(level=logging.INFO):
    log_colors_config = {
        "DEBUG": "white",  # cyan white
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }
    
    # 콘솔 출력 포매터
    console_formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s",
        datefmt="%Y-%m-%d  %H:%M:%S",
        log_colors=log_colors_config,
    )
    
    # 파일 출력 포매터
    file_formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s",
        datefmt="%Y-%m-%d  %H:%M:%S"
    )

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    
    # 파일 핸들러 (RotatingFileHandler 사용)
    file_handler = RotatingFileHandler(
        "tmp.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)

    # 핸들러 추가
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)