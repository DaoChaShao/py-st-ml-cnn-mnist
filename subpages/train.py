#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/24 13:12
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   train.py
# @Desc     :

from os import path, remove
from tensorflow.keras import metrics
from tensorflow.keras.utils import to_categorical
from streamlit import (empty, sidebar, subheader, session_state, button,
                       spinner, rerun, number_input, caption, columns,
                       slider, image, markdown)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

from utils.helper import Timer, TFKerasLogger

empty_messages: empty = empty()
empty_samp_title: empty = empty()
col_img, col_num = columns(2, gap="small")
empty_result_title: empty = empty()
col_los, col_acc, col_pre, col_rec, col_auc = columns(5, gap="small")
col_los_valid, col_acc_valid, col_pre_valid, col_rec_valid, col_auc_valid = columns(5, gap="small")
placeholder_los = col_los.empty()
placeholder_acc = col_acc.empty()
placeholder_pre = col_pre.empty()
placeholder_rec = col_rec.empty()
placeholder_auc = col_auc.empty()
placeholder_los_val = col_los_valid.empty()
placeholder_acc_val = col_acc_valid.empty()
placeholder_pre_val = col_pre_valid.empty()
placeholder_rec_val = col_rec_valid.empty()
placeholder_auc_val = col_auc_valid.empty()

pre_sessions: list[str] = ["X_train", "X_test", "y_train", "y_test"]
for session in pre_sessions:
    session_state.setdefault(session, None)
preprocess_sessions: list[str] = ["y_train_cat"]
for session in preprocess_sessions:
    session_state.setdefault(session, None)
model_sessions: list[str] = ["model", "histories", "mTimer"]
for session in model_sessions:
    session_state.setdefault(session, None)

MODEL_PATH: str = "mnist_model.h5"

with sidebar:
    if session_state["X_train"] is None:
        empty_messages.error("Please load the data on the Home page first.")
    else:

        subheader("Model Training Settings")

        if session_state["y_train_cat"] is None:
            empty_messages.info(
                f"Train Data is ready with {session_state['X_train'].shape}. Please preprocess the data first."
            )

            if button("Preprocess Train & Test Data", type="primary", width="stretch"):
                with spinner("Preprocessing Train & Test Data...", show_time=True, width="stretch"):
                    with Timer("Train & Test Data Preprocessing") as session_state["proTimer"]:
                        # One-hot encode the labels for metric callback, such as recall, precision etc.
                        session_state["y_train_cat"] = to_categorical(session_state["y_train"], num_classes=10)
                rerun()
        else:
            print(type(session_state["y_train_cat"]), session_state["y_train_cat"].shape)

            # Initialize the metrics placeholders
            placeholders: dict = {
                "loss": placeholder_los,
                "accuracy": placeholder_acc,
                "precision": placeholder_pre,
                "recall": placeholder_rec,
                "auc": placeholder_auc,
                "val_loss": placeholder_los_val,
                "val_accuracy": placeholder_acc_val,
                "val_precision": placeholder_pre_val,
                "val_recall": placeholder_rec_val,
                "val_auc": placeholder_auc_val
            }
            # Initialise the callback for visualisation
            callback = TFKerasLogger(placeholders)

            if session_state["model"] is None:
                empty_messages.info(f"Data is ready. You can start training the model.")

                validation_split = number_input(
                    "Validation Split",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.3,
                    step=0.1,
                    help="Fraction of the training data to be used as validation data.",
                )
                batch_size = number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=1024,
                    value=64,
                    step=1,
                    help="Number of samples per gradient update.",
                )
                epochs = number_input(
                    "Epochs",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=1,
                    help="Number of epochs to train the model.",
                )
                caption("Note: **64** batch size and **10** epochs are recommended for **QUICK** training.")

                if button("Train Mnist Model", type="primary", width="stretch"):
                    with spinner("Training Mnist Model...", show_time=True, width="stretch"):
                        with Timer("Mnist Model Training") as session_state["mTimer"]:
                            # Define the CNN model
                            session_state["model"] = Sequential([
                                # Input layer
                                Input(shape=(28, 28, 1)),
                                # First convolutional layer: 28 * 28 * 32
                                Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
                                # First max pooling layer: 14 * 14 * 32
                                MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
                                # Second convolutional layer: 14 * 14 * 64
                                Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
                                # Second max pooling layer: 7 * 7 * 64
                                MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
                                # Third convolutional layer: 7 * 7 * 128
                                Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
                                # Third max pooling layer: 7 * 7 * 128
                                MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
                                Flatten(),
                                Dense(128, activation="relu"),
                                Dense(10, activation="softmax"),
                            ])

                            session_state["model"].compile(
                                optimizer="adam",
                                # If the labels are one-hot encoded, use "categorical_crossentropy"
                                # If the labels are integers, use "sparse_categorical_crossentropy"
                                loss="categorical_crossentropy",
                                metrics=[
                                    "accuracy",
                                    metrics.Precision(name="precision"),
                                    metrics.Recall(name="recall"),
                                    metrics.AUC(name="auc"),
                                ],
                            )

                            # Train the model
                            session_state["histories"] = session_state["model"].fit(
                                session_state["X_train"],
                                session_state["y_train_cat"],
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=validation_split,
                                verbose=0,
                                callbacks=[callback],
                            )

                            # Get the training history for storage
                            session_state["histories"] = callback.get_history()
                    rerun()
            else:
                empty_result_title.markdown("### Model Training Result")
                hist = session_state["histories"]
                if hist:
                    last_epoch = len(hist["loss"])
                    for key, placeholder in placeholders.items():
                        if key in hist and placeholder is not None:
                            value = hist[key][-1]
                            label = f"Epoch {last_epoch}: {key.replace("val_", "valid ").capitalize()}"
                            placeholder.metric(label=label, value=f"{value:.4f}")

                if path.exists(MODEL_PATH):
                    empty_messages.info(
                        f"The model file **{MODEL_PATH}** already exists in the current directory."
                    )

                    if button("Delete the Model", type="secondary", width="stretch"):
                        with spinner("Deleting the model...", show_time=True, width="stretch"):
                            with Timer("Deleting the model") as timer:
                                remove(MODEL_PATH)

                                for placeholder in placeholders.values():
                                    placeholder.empty()

                                for session in model_sessions:
                                    session_state[session] = None
                        empty_messages.success(f"{timer} The model has been deleted successfully!")
                        rerun()
                else:
                    empty_messages.success(
                        f"{session_state["mTimer"]} The model has been trained successfully! You can save the model."
                    )

                    if button("Save the Model", type="primary", width="stretch"):
                        with spinner("Saving the model...", show_time=True, width="stretch"):
                            with Timer("Saving the model") as timer:
                                session_state["model"].save(MODEL_PATH)
                        empty_messages.success(f"{timer} The model has been saved successfully!")
                        rerun()

            if button("Clear Train & Test Data", type="secondary", width="stretch"):
                with spinner("Clearing Train & Test Data...", show_time=True, width="stretch"):
                    for session in preprocess_sessions:
                        session_state[session] = None
                rerun()
