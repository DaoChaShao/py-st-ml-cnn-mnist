<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**INTRODUCTION**
---
This application demonstrates how to train and evaluate a **Convolutional Neural Network (CNN)** on the classic **MNIST
handwritten digits dataset**, with an interactive interface built using **Streamlit**.  
Compared to a traditional Multi-Layer Perceptron (MLP), CNNs are better suited for image tasks, since convolutional
layers can automatically learn local features such as edges, textures, and shapes, leading to higher accuracy in image
recognition.

Within the app, users can:

- Visualize the training and validation process of a CNN model
- Explore the effects of different convolutional layers and parameter settings
- Draw digits in real-time and test the model’s inference ability

**DATA DESCRIPTION**
---
The **MNIST dataset** is one of the most widely used benchmark datasets in machine learning and computer vision. Key
properties include:

- **Dataset size**
    - Training set: 60,000 grayscale images of size 28×28
    - Test set: 10,000 grayscale images of size 28×28

- **Image content**
    - Each image represents a handwritten digit from 0 to 9
    - Images are **single-channel grayscale**, with pixel values ranging from 0 to 255

- **Task objective**
    - Input: a 28×28 grayscale handwritten digit image
    - Output: predict the corresponding digit class (0–9)

This project helps users **understand the advantages of CNNs in image classification** and provides a hands-on
comparison between CNNs and MLPs on the MNIST dataset.

**FEATURES**
---

- **Data Loading & Preprocessing:** Load MNIST dataset and preprocess for MLP training (flattening and normalization).
- **Model Training:** Train a Multi-Layer Perceptron with configurable epochs, batch size, and validation split.
- **Real-time Training Metrics:** Monitor loss, accuracy, precision, recall, and AUC for both training and validation
  sets.
- **Model Testing:** Evaluate model performance with accuracy and R² score on the test set.
- **Real-time Digit Recognition:** Draw digits on a canvas and get immediate predictions using the trained model.
- **Visualization Tools:** Scatter plots and decision boundary visualization for 2D/3D datasets (for experiments beyond
  MNIST).

**QUICK START**
---

1. Clone the repository to your local machine.
2. Install the required dependencies with the command `pip install -r requirements.txt`.
3. Run the application with the command `streamlit run main.py`.
4. You can also try the application by visiting the following
   link:  
   [![Static Badge](https://img.shields.io/badge/Open%20in%20Streamlit-Daochashao-red?style=for-the-badge&logo=streamlit&labelColor=white)](https://cnn-mnist.streamlit.app/)

**WEB DEVELOPMENT**
---

1. Install NiceGUI with the command `pip install streamlit`.
2. Run the command `pip show streamlit` or `pip show streamlit | grep Version` to check whether the package has been
   installed and its version.
3. Run the command `streamlit run app.py` to start the web application.

**PRIVACY NOTICE**
---
This application may require inputting personal information or private data to generate customised suggestions,
recommendations, and necessary results. However, please rest assured that the application does **NOT** collect, store,
or transmit your personal information. All processing occurs locally in the browser or runtime environment, and **NO**
data is sent to any external server or third-party service. The entire codebase is open and transparent — you are
welcome to review the code [here](./) at any time to verify how your data is handled.

**LICENCE**
---
This application is licensed under the [BSD-3-Clause License](LICENSE). You can click the link to read the licence.

**CHANGELOG**
---
This guide outlines the steps to automatically generate and maintain a project changelog using git-changelog.

1. Install the required dependencies with the command `pip install git-changelog`.
2. Run the command `pip show git-changelog` or `pip show git-changelog | grep Version` to check whether the changelog
   package has been installed and its version.
3. Prepare the configuration file of `pyproject.toml` at the root of the file.
4. The changelog style is [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
5. Run the command `git-changelog`, creating the `Changelog.md` file.
6. Add the file `Changelog.md` to version control with the command `git add Changelog.md` or using the UI interface.
7. Run the command `git-changelog --output CHANGELOG.md` committing the changes and updating the changelog.
8. Push the changes to the remote repository with the command `git push origin main` or using the UI interface.