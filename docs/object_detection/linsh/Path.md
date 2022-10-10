## 说明

Path的使用说明，可以做到与系统无关

```python
from pathlib import Path

FILE = Path(__file__).resolve()
print(FILE) # <PosixPath.parents> 
print(FILE.parents) # <PosixPath.parents> 
for x in FILE.parents:
    print(x)

print(FILE.parents[1])
```

在目录`/Users/cuidonglin/Documents/myyolov5/interproject/`下运行
python test.py

返回结果如下：
```
/Users/cuidonglin/Documents/myyolov5/interproject/test.py
<PosixPath.parents> 

/Users/cuidonglin/Documents/myyolov5/interproject [0]
/Users/cuidonglin/Documents/myyolov5 [1]
/Users/cuidonglin/Documents [2]
/Users/cuidonglin [3]
/Users [4]
/

/Users/cuidonglin/Documents/myyolov5  [1]
```

## 小命令

1. 判断文件是否存在：`file.exists()` # 其中 file 是Path对象即可

2. 打开文件：`with file.open() as f:`

3. 创建文件夹：
```
tmp = Path(ROOT/'tmp')
tmp.mkdir(parents=True, exist_ok=True) # exist_ok=True可以使得tmp已经存在的情况下不报错，也不覆盖原文件夹
``` 

4. 文件的更新时间

```python
from datetime import datetime
t = datetime.fromtimestamp(Path(ROOT/'test.py').stat().st_mtime)
print(f'{t.year}-{t.month}-{t.day}')
```

