#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/24 13:12
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   test.py
# @Desc     :

from os import path
from sklearn.metrics import accuracy_score, r2_score
from streamlit import (empty, sidebar, subheader, session_state, button,
                       spinner, rerun, columns, metric, slider,
                       caption, image, markdown)
from tensorflow.keras.models import load_model

from subpages.train import MODEL_PATH
from utils.helper import Timer

empty_messages: empty = empty()
empty_samp_title: empty = empty()
col_img, col_num = columns(2, gap="small")
empty_result_title: empty = empty()
col_acc, col_r2 = columns(2, gap="small")

pre_sessions: list[str] = ["X_train", "X_test", "y_train", "y_test"]
for session in pre_sessions:
    session_state.setdefault(session, None)
preprocess_sessions: list[str] = ["y_train_cat"]
for session in preprocess_sessions:
    session_state.setdefault(session, None)
test_sessions: list[str] = ["tTimer", "y_pred"]
for session in test_sessions:
    session_state.setdefault(session, None)

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

                if session_state["y_pred"] is None:
                    empty_messages.info("Model is ready. Please test the model first.")

                    if button("Test the Model", type="primary", width="stretch"):
                        with spinner("Testing the Model...", show_time=True, width="stretch"):
                            with Timer("Model Testing") as session_state["tTimer"]:
                                model = load_model(MODEL_PATH)
                                # Make predictions
                                y_pred_probabilities = model.predict(session_state["X_test"])
                                session_state["y_pred"] = y_pred_probabilities.argmax(axis=1)

                        rerun()
                else:
                    empty_messages.success(f"{session_state['tTimer']} Model has been tested.")

                    empty_result_title.markdown("#### Test Results")
                    acc = accuracy_score(session_state["y_test"], session_state["y_pred"])
                    r2 = r2_score(session_state["y_test"], session_state["y_pred"])
                    with col_acc:
                        metric("Accuracy", f"{acc:.4%}")
                    with col_r2:
                        metric("R² Score", f"{r2:.4f}")

                    index_test = slider(
                        "Select the index of the test sample to display",
                        min_value=0,
                        max_value=len(session_state["X_test"]) - 1,
                        value=27,
                        step=1,
                        help="Select the index of the test sample to display",
                    )
                    caption(f"The maximum number of test samples is **{len(session_state['X_test'])}**.")

                    if button("Predict the Selected Sample", type="primary", width="stretch"):
                        with spinner("Predicting the Selected Sample...", show_time=True, width="stretch"):
                            with col_img:
                                empty_samp_title.markdown(f"### Test Sample at Index {index_test}")
                                print(type(session_state["X_test"]), session_state["X_test"].shape, index_test)
                                image(
                                    session_state["X_test"][index_test],
                                    caption=f"**True Label: {session_state['y_test'][index_test]}**",
                                    width="stretch"
                                )
                            with col_num:
                                pred_label = session_state["y_pred"][index_test]
                                markdown(
                                    f"<h1 style='font-size:300px; font-weight:bold; text-align:center;'>{pred_label}</h1>",
                                    unsafe_allow_html=True, width="stretch"
                                )

                    if button("Retest the Test", type="secondary", width="stretch"):
                        for session in test_sessions:
                            session_state[session] = None
                        rerun()
