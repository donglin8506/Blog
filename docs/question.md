## 说明

记录存在疑问的点

## 环境相关的问题

1. `os.getenv('RANK', -1)` ?
   随便打开一个脚本，运行该命令，返回-1，即 os.getenv('RANK')返回None；应该是多GPU环境中会使用，暂时不会使用； rank的值可能是 -1或者0


2. logging 中的handler是什么含义
3. 