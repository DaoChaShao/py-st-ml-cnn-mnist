#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/24 13:12
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   realtime.py
# @Desc     :

from PIL import Image
from numpy import array
from os import path, remove
from streamlit import (empty, sidebar, subheader, session_state, button,
                       spinner, rerun, slider, selectbox, columns, image)
from streamlit_drawable_canvas import st_canvas
from subpages.train import MODEL_PATH
from tensorflow.keras.models import load_model

from utils.helper import Timer

empty_messages: empty = empty()
empty_write_title: empty = empty()
col_write, col_small = columns(2, gap="small")
empty_pred_result: empty = empty()

pre_sessions: list[str] = ["X_train", "X_test", "y_train", "y_test"]
for session in pre_sessions:
    session_state.setdefault(session, None)
preprocess_sessions: list[str] = ["y_train_cat"]
for session in preprocess_sessions:
    session_state.setdefault(session, None)
test_sessions: list[str] = ["rtTimer", "img_input", "y_prediction"]
for session in test_sessions:
    session_state.setdefault(session, None)

with sidebar:
    with sidebar:
        if session_state["X_train"] is None:
            empty_messages.error("Please load the data on the Home page first.")
        else:
            if session_state["y_train_cat"] is None:
                empty_messages.error("Please preprocess the data on the Data Preparation page first.")
            else:
                if not path.exists(MODEL_PATH):
                    empty_messages.error("Please train the model on the Model Training page and save it first.")
                else:
                    subheader("Model Testing Settings")

                    drawing_modes: list[str] = ["point", "freedraw", "line", "rect", "circle", "transform"]
                    drawing_mode: str = selectbox(
                        "Drawing tool: ",
                        options=drawing_modes,
                        index=1,
                        disabled=True,
                        help="Select the drawing tool",
                    )

                    stroke_width = slider(
                        "Stroke width: ",
                        min_value=5,
                        max_value=30,
                        value=15,
                        step=1,
                        help="Width of the stroke",
                    )

                    canvas_size: int = 420
                    zoom_scale: int = int(420 / 28)

                    empty_write_title.markdown("#### Real-time Digit Recognition")
                    with col_write:
                        canvas_result = st_canvas(
                            stroke_width=stroke_width,
                            stroke_color="white",
                            background_color="black",
                            update_streamlit=True,
                            height=canvas_size,
                            width=canvas_size,
                            drawing_mode=drawing_mode,
                        )

                    if canvas_result.image_data is not None:
                        # The image data should in RGBA format, such as uint8
                        print(type(canvas_result.image_data), canvas_result.image_data.dtype)
                        print(canvas_result.image_data.min(), canvas_result.image_data.max())
                        print(canvas_result.image_data.shape)
                        # Convert the image data to a PIL Image
                        img = Image.fromarray(canvas_result.image_data)
                        # Convert the color image to grayscale
                        # img_gray = img[:, :, 0]
                        img_gray = img.convert("L")
                        # Resize the image to 28x28
                        zoom_size: int = canvas_size // zoom_scale
                        img_small = img_gray.resize((zoom_size, zoom_size), resample=Image.Resampling.LANCZOS)
                        print(type(img_small), img_small.size)
                        # Convert the image to a numpy array because the model input is numpy array
                        img_array = array(img_small)
                        print(type(img_array), img_array.dtype)
                        print(img_array.min(), img_array.max())
                        print(img_array.shape)
                        # Normalize the image to [0, 1] because the model expects input as normalized float32
                        img_normalised = img_array.astype("float32")
                        print(type(img_normalised), img_normalised.dtype)
                        print(img_normalised.min(), img_normalised.max())
                        print(img_normalised.shape)
                        # Reshape the image to (1, 28, 28, 1) because the model expects input shape as (batch_size, height, width, channels)
                        input_batch_size: int = 1
                        input_img_size: tuple = (28, 28, 1)
                        session_state["img_input"] = img_normalised.reshape((input_batch_size,) + input_img_size)

                    with col_small:
                        image(
                            img_small,
                            caption="The zoom-out image after resizing to 28x28",
                            width="stretch"
                        )

                    if session_state["y_prediction"] is None:
                        empty_messages.warning("Model is ready. Please draw a digit (0-9) on the canvas.")

                        if button("Predict the Number", type="primary", width="stretch"):
                            with spinner("Predicting the Number...", show_time=True, width="stretch"):
                                with Timer("Real-time Prediction") as session_state["rtTimer"]:
                                    model = load_model(MODEL_PATH)
                                    # Make predictions
                                    y_pred_probabilities = model.predict(session_state["img_input"])
                                    session_state["y_prediction"] = y_pred_probabilities.argmax(axis=1)[0]
                                    print(y_pred_probabilities)
                            rerun()
                    else:
                        empty_messages.success(f"{session_state["rtTimer"]} Prediction completed.")
                        empty_pred_result.markdown(f"🧠 Predicted Number: **{session_state["y_prediction"]}**")

                        if button("Clear Drawing Canvas", type="secondary", width="stretch"):
                            canvas_result.image_data = None
                            for session in test_sessions:
                                session_state[session] = None
                            rerun()

                        SAVE_PATH: str = "canvas_image.png"
                        if not path.exists(SAVE_PATH):
                            if button("Save your Drawing as Image", type="primary", width="stretch"):
                                img.save(SAVE_PATH)
                                empty_messages.success(f"Your drawing has been saved as {SAVE_PATH}.")
                                rerun()
                        else:
                            if button("Delete the Saved Image", type="secondary", width="stretch"):
                                remove(SAVE_PATH)
                                empty_messages.success(f"The saved image {SAVE_PATH} has been deleted.")
                                rerun()
