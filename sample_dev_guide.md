# 未来科技 - 后端开发环境搭建指南 (2025)

## 1. 简介
本文档旨在帮助新入职的后端工程师快速配置开发环境，确保能够顺利运行 `zhishikuer` 项目。

## 2. 必备工具
请确保你的电脑已经安装了以下基础软件：

*   **操作系统**: Windows 10/11 或 macOS 13+
*   **Git**: 版本 >= 2.30
*   **Python**: 版本 **3.10** (推荐) 或 3.11

## 3. Python 环境配置

### 3.1 安装 Python
推荐使用 Anaconda 或 Miniconda 管理 Python 环境。

```bash
# 创建虚拟环境
conda create -n zhishikuer python=3.10
# 激活环境
conda activate zhishikuer
```

### 3.2 安装依赖
项目依赖分为 CPU 版和 GPU 版，请根据硬件选择。

#### CPU 用户 (通用)
```bash
pip install -r requirements.txt
```

#### GPU 用户 (NVIDIA 显卡)
如果你的显卡显存 >= 6GB，可以使用 GPU 加速：
```bash
pip install -r requirements_gpu.txt
```
> **注意**: GPU 版本需要提前安装 CUDA Toolkit 11.8 或更高版本。

## 4. 运行项目
在项目根目录下运行以下命令启动服务：

```bash
# 启动后端服务
python main.py
```
服务启动后，浏览器访问 `http://localhost:8000` 即可看到主界面。

## 5. 常见问题 (FAQ)

### Q: 为什么安装依赖时报错 `llama-cpp-python` 编译失败？
**A**: Windows 用户通常缺少 C++ 编译环境。请安装 Visual Studio 2019/2022 并勾选 "C++ 桌面开发" 组件。或者直接下载预编译的 `.whl` 文件安装。

### Q: 知识库支持哪些文件格式？
**A**: 目前支持 `.txt`, `.pdf`, `.md`, `.html` 格式的文件。

---
*最后更新时间: 2025-02-21 by 技术部*
