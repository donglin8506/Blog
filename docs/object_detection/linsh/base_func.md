## 说明

本篇主要记录跟不再调用其他函数的原子函数

## 1. `colorstr`函数

出现的文件：

```python

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

if __name__ == "__main__":
    prefix = colorstr('red', 'bold', 'requirements:')
    print(prefix)
```

在终端看到的 字符串 "requirements:"是红色、粗体

## 2. python环境检查

```python
import pkg_resources as pkg
import platform

print(platform.python_version()) # 获取当前使用的python版本 

pkg.parse_version(x) # 不确定什么意思
# 可以获取需要安装的包
with file.open() as f:
    pkg.parse_requirements(f) # f是 Path('requirements.txt)

```

[python setuptools之pkg_resources模块](https://gimefi.gitbooks.io/python-pkg_resources/content/)


## 3, 检查是否是kaggle环境 

```python 
def is_kaggle():
    # Is environment a Kaggle Notebook?
    try:
        assert os.environ.get('PWD') == '/kaggle/working'
        assert os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'
        return True
    except AssertionError:
        return False
```

获取当前路径：`os.environ.get('PWD')`

断言不存在问题返回True，有问题返回False: `try except AssertionError`