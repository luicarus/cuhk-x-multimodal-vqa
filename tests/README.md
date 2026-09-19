# CPU 测试

```powershell
python scripts/test.py -q
```

使用当前 Python 3.11 解释器运行 pytest，禁止 bytecode 和 pytest 缓存；所有 fixture、子进程临时文件和模拟运行结果放在系统临时目录，结束后清除。只在终端显示结果，不写阶段验收文档。

测试不执行云端 Notebook 的 cells，不联网、不加载 GPU、不抽帧。Notebook 保留原执行记录，测试只读取语法、CLI 调用兼容性和打包后的字节一致性。
