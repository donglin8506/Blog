## 说明

本篇主要记录跟日志相关的内容


## 小命令

```python
import logging

print(logging.WARNING)  # 30 不知道为什么是30
print(logging.INFO) # 20
```

## 1. logging使用

```python
import logging
import os

print(logging.WARNING)
print(logging.INFO)
def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    log.addHandler(handler)

set_logging()
LOGGER = logging.getLogger("yolov5")
name = 'python'
minimum = '3.7.0'
current = '3.7.11'
s = f'{name}{minimum} required by YOLOv5, but {name}{current} is currently installed'  # string
LOGGER.warning(s)
```

暂时不懂