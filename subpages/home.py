#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/24 13:11
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :

from streamlit import title, expander, caption, empty

empty_message = empty()
empty_message.info("Please check the details at the different pages of core functions.")

title("Convolutional Neural Network (CNN) for MNIST Digit Classification")
with expander("**INTRODUCTION**", expanded=True):
    caption("+ 📂 Load MNIST dataset and preprocess for model training.")
    caption("+ 🧠 Train a Convolutional Neural Network with custom epochs and batch size.")
    caption("+ 📊 Visualize training metrics in real-time (loss, accuracy, precision, recall, AUC).")
    caption("+ 🧪 Test the trained model on the MNIST test dataset.")
    caption("+ ✏️ Draw digits on a canvas and get instant predictions.")
