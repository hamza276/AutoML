import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
     mean_squared_error,
     mean_absolute_error,
     classification_report
)

def evaluate_model(trained_models, X_test, y_test, task):
   for i, (model_obj, model_label) in enumerate(trained_models):
        if model_obj is None:
            continue
        try:
            if hasattr(model_obj, 'best_estimator_'):
                best_mod = model_obj.best_estimator_
                y_pred = best_mod.predict(X_test)
            else:
                y_pred = model_obj.predict(X_test)
        except Exception as e:
            st.error(f"Error generating predictions for model {i+1}: {e}")
            continue

        st.subheader(f"Model {i+1} [{model_label}] Test Results")
        if task == 'Classification':
            st.write(classification_report(y_test, y_pred))
        else:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            st.write(f"Mean Squared Error (MSE): {mse:.6f}")
            st.write(f"Mean Absolute Error (MAE): {mae:.6f}")


def compare_models(trained_models, X_test, y_test, model_info):
    if trained_models and X_test is not None:
        # By default, select all
        model_options = [
            f"Model {i+1} [{label}]" for i, (_, label) in enumerate(trained_models)
        ]
        selected_models = st.multiselect(
            "Select Models for Comparison",
            options=model_options,
            default=model_options
        )

        # We'll keep a dictionary that maps "comparison row name" -> (model_index_in_state)
        # so we don't rely on complicated string parsing to find the best model
        name_to_index_map = {}
        comparison_metrics = {}

        for sm in selected_models:
            # e.g. "Model 1 [No Tuning(Default)]"
            idx_str = sm.split(" ")[1]  # '1'
            idx_int = int(idx_str) - 1
            
