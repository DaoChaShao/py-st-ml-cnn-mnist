<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**应用简介**
---
本应用展示了如何使用 **卷积神经网络（CNN）** 对经典的 **MNIST 手写数字数据集**进行训练与预测，并通过 **Streamlit**
提供可交互的学习体验。  
与传统的多层感知机（MLP）相比，CNN 更适合处理图像任务，因为卷积层能够自动提取局部特征（如边缘、纹理、形状），从而在图像识别中表现更优。

用户可以在应用中：

- 观察 CNN 模型的训练与验证过程
- 查看不同卷积层的作用与参数设置
- 实时绘制数字并进行预测，体验模型的推理能力

**数据描述**
---
**MNIST 数据集**是机器学习和计算机视觉领域的入门经典数据集。其主要特点如下：

- **数据规模**
    - 训练集：60,000 张 28×28 的灰度图像
    - 测试集：10,000 张 28×28 的灰度图像

- **图像内容**
    - 每张图像表示一个手写数字（0–9）
    - 图像为 **单通道灰度图**，像素值范围 0–255

- **任务目标**
    - 输入：一张 28×28 的手写数字图像
    - 输出：预测该图像对应的数字类别（0–9）

通过该应用，用户可以直观地理解 **CNN 在图像分类中的优势**，并比较 CNN 与 MLP 在 MNIST 数据集上的表现差异。

**功能特性**
---

- **数据加载与预处理：** 加载 MNIST 数据集并进行扁平化、归一化处理。
- **模型训练：** 支持自定义训练轮数、批量大小和验证集比例的多层感知机训练。
- **实时训练指标：** 可实时监控训练与验证集的损失、准确率、精确率、召回率和 AUC。
- **模型测试：** 在测试集上评估模型性能，提供准确率和 R² 值。
- **实时数字识别：** 在画布上绘制数字并使用训练好的模型进行即时预测。
- **可视化工具：** 支持二维/三维散点图与决策边界可视化（可用于 MNIST 以外的实验数据）。

**快速开始**
---

1. 将本仓库克隆到本地计算机。
2. 使用以下命令安装所需依赖项：`pip install -r requirements.txt`
3. 使用以下命令运行应用程序：`streamlit run main.py`
4. 你也可以通过点击以下链接在线体验该应用：  
   [![Static Badge](https://img.shields.io/badge/Open%20in%20Streamlit-Daochashao-red?style=for-the-badge&logo=streamlit&labelColor=white)](https://cnn-mnist.streamlit.app/)

**网页开发**
---

1. 使用命令`pip install streamlit`安装`Streamlit`平台。
2. 执行`pip show streamlit`或者`pip show git-streamlit | grep Version`检查是否已正确安装该包及其版本。
3. 执行命令`streamlit run app.py`启动网页应用。

**隐私声明**
---
本应用可能需要您输入个人信息或隐私数据，以生成定制建议和结果。但请放心，应用程序 **不会**
收集、存储或传输您的任何个人信息。所有计算和数据处理均在本地浏览器或运行环境中完成，**不会** 向任何外部服务器或第三方服务发送数据。

整个代码库是开放透明的，您可以随时查看 [这里](./) 的代码，以验证您的数据处理方式。

**许可协议**
---
本应用基于 **BSD-3-Clause 许可证** 开源发布。您可以点击链接阅读完整协议内容：👉 [BSD-3-Clause License](./LICENSE)。

**更新日志**
---
本指南概述了如何使用 git-changelog 自动生成并维护项目的变更日志的步骤。

1. 使用命令`pip install git-changelog`安装所需依赖项。
2. 执行`pip show git-changelog`或者`pip show git-changelog | grep Version`检查是否已正确安装该包及其版本。
3. 在项目根目录下准备`pyproject.toml`配置文件。
4. 更新日志遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/) 提交规范。
5. 执行命令`git-changelog`创建`Changelog.md`文件。
6. 使用`git add Changelog.md`或图形界面将该文件添加到版本控制中。
7. 执行`git-changelog --output CHANGELOG.md`提交变更并更新日志。
8. 使用`git push origin main`或 UI 工具将变更推送至远程仓库。
