import streamlit as st
import pandas as pd
import pickle
from AutoML_app.utils.data_handler import (
    display_dataset,
    handle_missing_values,
    split_data,
    handle_categorical_variables
)
from AutoML_app.utils.eda import exploratory_data_analysis
from AutoML_app.utils.model_trainer import (
     train_model_with_grid_search,
     train_model_with_random_search,
     train_model_with_optuna,
     get_model_class
)
from AutoML_app.utils.model_evaluation import (
     evaluate_model,
     compare_models
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

def main():
    # -- Keep This Title Stationary
    st.title("AutoML Application with Multiple Fine-Tuning Methods")

    # ------------------------------------------
    # Initialize session state variables
    # ------------------------------------------
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = []  # (model_obj, label_str)
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'model_info' not in st.session_state:
        st.session_state.model_info = {}
    if 'fine_tuning_triggered' not in st.session_state:
        st.session_state.fine_tuning_triggered = False
    if 'fine_tuned_model' not in st.session_state:
        st.session_state.fine_tuned_model = None

    # ------------------------------------------
    # Sidebar
    # ------------------------------------------
    with st.sidebar:
        st.header("Preferences")
        file_type = st.radio("Select file type", options=["CSV", "Excel (XLSX)"])
        enable_eda = st.checkbox("Perform EDA")
        enable_description = st.checkbox("Show Descriptive Statistics")
        task = st.selectbox("Select Task", ['Classification', 'Regression'])
        missing_value_strategy = st.selectbox("Handle Missing Values", ['None', 'Mean', 'Median', 'Interpolation'])

        tuning_approach = st.radio(
            "Choose your hyperparameter tuning approach:",
            ["No Tuning (Defaults)", "Manual Hyperparams", "Complete Fine-Tuning"],
            index=0
        )
        finetuning_library = None
        if tuning_approach == "Complete Fine-Tuning":
            finetuning_library = st.selectbox(
                "Select Library for Hyperparameter Tuning",
                ["Optuna", "Grid Search", "Random Search"]
            )

        classification_models = [
            'Logistic Regression',
            'Decision Tree Classifier',
            'Random Forest Classifier',
            'SVM Classifier',
            'MLP Classifier',
            'KNeighbors Classifier',
            'GaussianNB',
            'AdaBoost Classifier',
            'GradientBoosting Classifier',
            'ExtraTrees Classifier'
        ]
        regression_models = [
            'Decision Tree Regressor',
            'Random Forest Regressor',
            'SVM Regressor',
            'MLP Regressor',
            'Linear Regression',
            'Ridge Regression',
            'Lasso Regression',
            'ElasticNet Regression',
            'KNeighbors Regressor',
            'AdaBoost Regressor'
        ]

        model_type = st.selectbox(
            "Select a Model",
            classification_models if task == 'Classification' else regression_models
        )

        # If Manual Hyperparams -> Show Sliders
        hyperparameters = {}
        if tuning_approach == "Manual Hyperparams":
            st.markdown("**Manual Hyperparameter Selection:**")
            if model_type == 'Logistic Regression':
                hyperparameters['C'] = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
                hyperparameters['max_iter'] = st.slider("Max Iterations", 100, 2000, 1000)
            elif model_type == 'Decision Tree Classifier':
                hyperparameters['max_depth'] = st.slider("Max Depth", 1, 20, 5)
                hyperparameters['min_samples_split'] = st.slider("Min Samples Split", 2, 10, 2)
            elif model_type == 'Random Forest Classifier':
                hyperparameters['n_estimators'] = st.slider("Number of Estimators", 10, 300, 100)
                hyperparameters['max_depth'] = st.slider("Max Depth", 1, 20, 5)
            elif model_type == 'SVM Classifier':
                hyperparameters['C'] = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
                hyperparameters['kernel'] = st.selectbox("Kernel", ['linear', 'rbf', 'poly'])
            elif model_type == 'MLP Classifier':
                hyperparameters['hidden_layer_sizes'] = st.slider("Hidden Layer Size", 10, 200, 100)
                hyperparameters['max_iter'] = st.slider("Max Iterations", 100, 2000, 1000)
            elif model_type == 'KNeighbors Classifier':
                hyperparameters['n_neighbors'] = st.slider("n_neighbors", 1, 15, 5)
                hyperparameters['weights'] = st.selectbox("Weights", ['uniform', 'distance'])
            elif model_type == 'GaussianNB':
                st.write("No typical manual hyperparameters for GaussianNB.")
            elif model_type == 'AdaBoost Classifier':
                hyperparameters['n_estimators'] = st.slider("Number of Estimators", 10, 200, 50)
                hyperparameters['learning_rate'] = st.slider("Learning Rate", 0.01, 2.0, 1.0)
            elif model_type == 'GradientBoosting Classifier':
                hyperparameters['n_estimators'] = st.slider("Number of Estimators", 10, 200, 50)
                hyperparameters['learning_rate'] = st.slider("Learning Rate", 0.01, 1.0, 0.1)
                hyperparameters['max_depth'] = st.slider("Max Depth", 1, 10, 3)
            elif model_type == 'ExtraTrees Classifier':
                hyperparameters['n_estimators'] = st.slider("Number of Estimators", 10, 200, 100)
                hyperparameters['max_depth'] = st.slider("Max Depth", 1, 20, 5)

            elif model_type == 'Decision Tree Regressor':
                hyperparameters['max_depth'] = st.slider("Max Depth", 1, 20, 5)
                hyperparameters['min_samples_split'] = st.slider("Min Samples Split", 2, 10, 2)
            elif model_type == 'Random Forest Regressor':
                hyperparameters['n_estimators'] = st.slider("Number of Estimators", 10, 300, 100)
                hyperparameters['max_depth'] = st.slider("Max Depth", 1, 20, 5)
            elif model_type == 'SVM Regressor':
                hyperparameters['C'] = st.slider("C (Regularization)", 0.01, 10.0, 1.0)
                hyperparameters['kernel'] = st.selectbox("Kernel", ['linear', 'rbf', 'poly'])
            elif model_type == 'MLP Regressor':
                hyperparameters['hidden_layer_sizes'] = st.slider("Hidden Layer Size", 10, 200, 100)
                hyperparameters['max_iter'] = st.slider("Max Iterations", 100, 2000, 1000)
            elif model_type == 'Linear Regression':
                st.write("No typical manual hyperparameters for LinearRegression.")
            elif model_type == 'Ridge Regression':
                hyperparameters['alpha'] = st.slider("Alpha", 0.01, 10.0, 1.0)
            elif model_type == 'Lasso Regression':
                hyperparameters['alpha'] = st.slider("Alpha", 0.01, 10.0, 1.0)
            elif model_type == 'ElasticNet Regression':
                hyperparameters['alpha'] = st.slider("Alpha", 0.01, 10.0, 1.0)
                hyperparameters['l1_ratio'] = st.slider("L1 Ratio", 0.0, 1.0, 0.5)
            elif model_type == 'KNeighbors Regressor':
                hyperparameters['n_neighbors'] = st.slider("n_neighbors", 1, 15, 5)
                hyperparameters['weights'] = st.selectbox("Weights", ['uniform', 'distance'])
            elif model_type == 'AdaBoost Regressor':
                hyperparameters['n_estimators'] = st.slider("Number of Estimators", 10, 200, 50)
                hyperparameters['learning_rate'] = st.slider("Learning Rate", 0.01, 2.0, 1.0)

        train_button = st.button("Train Model")
        test_button = st.button("Test Model")

    # ------------------------------------------
    # File Upload
    # ------------------------------------------
    st.subheader("Upload your dataset")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Load file
        try:
            if file_type == "CSV":
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            else:
                df = pd.read_excel(uploaded_file)
        except UnicodeDecodeError:
            st.warning("UTF-8 encoding failed. Retrying with 'latin1' encoding...")
            try:
                if file_type == "CSV":
                    df = pd.read_csv(uploaded_file, encoding='latin1')
                else:
                    df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Failed to load the file: {e}")
                return
        except pd.errors.ParserError as e:
            st.error(f"Parsing error: {e}")
            return

        st.write("Dataset loaded successfully!")
        display_dataset(df)

        if enable_description:
            st.subheader("Descriptive Statistics")
            st.write(df.describe())

        if enable_eda:
            st.subheader("Exploratory Data Analysis")
            exploratory_data_analysis(df)

        # Target selection
        target_variable = st.selectbox("Select Target Variable", df.columns)
        if target_variable:
            X, y = split_data(df, target_variable)
            X, y = handle_categorical_variables(X, y, task)

            if missing_value_strategy != 'None':
                X = handle_missing_values(X, missing_value_strategy)
                y = y[X.index]

            if X.isnull().values.any() or y.isnull().values.any():
                st.error("Missing values remain after preprocessing. Dropping missing rows.")
                X = X.dropna()
                y = y[X.index]

            if len(X) != len(y):
                st.error("Inconsistent samples between features and target.")
                return

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # -------------------------------------------------
            # TRAIN MODEL
            # -------------------------------------------------
            if train_button:
                model_class = get_model_class(model_type)
                if model_class is None:
                    st.error("Unsupported model selected.")
                    return

                model_obj = None

                # 1) No Tuning (Defaults)
                if tuning_approach == "No Tuning (Defaults)":
                    try:
                        model_obj = model_class()
                        st.info(f"Training {model_type} with **default** hyperparameters.")
                        model_obj.fit(X_train, y_train)
                        label_str = "No Tuning(Default)"
                    except Exception as e:
                        st.error(f"Failed to train with default hyperparams: {e}")
                        return

                # 2) Manual Hyperparams
                elif tuning_approach == "Manual Hyperparams":
                    try:
                        model_obj = model_class(**hyperparameters)
                        st.info(f"Training {model_type} with manual hyperparams: {hyperparameters}")
                        model_obj.fit(X_train, y_train)
                        label_str = "Manual-Hyperparams"
                    except Exception as e:
                        st.error(f"Failed to instantiate/train model with manual hyperparams: {e}")
                        return

                # 3) Complete Fine-Tuning
                else:  
                    if finetuning_library == "Grid Search":
                        param_grid = {}
                        if model_type == 'Random Forest Classifier':
                            param_grid = {
                                'n_estimators': [50, 100],
                                'max_depth': [3, 5]
                            }
                        elif model_type == 'SVM Regressor':
                            param_grid = {
                                'C': [0.1, 1, 10],
                                'kernel': ['linear', 'rbf']
                            }
                        else:
                            if hasattr(model_class(), 'max_iter'):
                                param_grid = {'max_iter': [100, 500]}
                            else:
                                param_grid = {'dummy_param': [True, False]}

                        st.info(f"Performing complete fine-tuning with **Grid Search** on {model_type}.")
                        model_obj = train_model_with_grid_search(model_class(), param_grid, X_train, y_train)
                        label_str = "Complete-FineTuning(GridSearch)"

                    elif finetuning_library == "Random Search":
                        param_dist = {}
                        if model_type == 'Random Forest Regressor':
                            param_dist = {
                                'n_estimators': [10, 50, 100],
                                'max_depth': [3, 5, 7]
                            }
                        elif model_type == 'SVM Classifier':
                            param_dist = {
                                'C': [0.01, 0.1, 1, 10],
                                'kernel': ['linear', 'rbf']
                            }
                        else:
                            if hasattr(model_class(), 'max_iter'):
                                param_dist = {
                                    'max_iter': [100, 200, 500]
                                }
                            else:
                                param_dist = {
                                    'dummy_param': [True, False]
                                }

                        st.info(f"Performing complete fine-tuning with **Random Search** on {model_type}.")
                        model_obj = train_model_with_random_search(model_class(), param_dist, X_train, y_train)
                        label_str = "Complete-FineTuning(RandomSearch)"

                    elif finetuning_library == "Optuna":
                        st.info(f"Performing complete fine-tuning with **Optuna** on {model_type}.")
                        model_obj = train_model_with_optuna(model_class, X_train, y_train, task, model_type)
                        label_str = "Complete-FineTuning(Optuna)"
                    else:
                        st.error("Unknown library selected for fine-tuning.")
                        return

                # Save model in session
                if model_obj:
                    st.session_state.trained_models.append((model_obj, label_str))
                    st.session_state.model_info = {
                        'task': task,
                        'model_type': model_type,
                        'tuning_approach': tuning_approach,
                        'finetuning_library': finetuning_library
                    }
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test

                    # Let user download
                    st.subheader("Training Completed")
                    if hasattr(model_obj, 'best_estimator_'):
                        best_est = model_obj.best_estimator_
                        st.write("Best Hyperparameters Found:", model_obj.best_params_)
                        download_data = pickle.dumps(model_obj)
                        if st.button("Download Trained Model"):
                            st.download_button(
                                label="Download Model File",
                                data=download_data,
                                file_name=f"{model_type.replace(' ','_')}_complete_finetuning.pkl"
                            )
                    else:
                        download_data = pickle.dumps(model_obj)
                        if st.button("Download Trained Model"):
                            st.download_button(
                                label="Download Model File",
                                data=download_data,
                                file_name=f"{model_type.replace(' ','_')}_{tuning_approach}.pkl"
                            )

            # -------------------------------------------------
            # TEST MODEL
            # -------------------------------------------------
            if test_button:
                if st.session_state.trained_models and st.session_state.X_test is not None:
                    evaluate_model(st.session_state.trained_models, st.session_state.X_test, st.session_state.y_test, task)
                else:
                    st.error("No models or no test data. Please train a model first.")

    # -------------------------------------------------
    # MODEL COMPARISON
    # -------------------------------------------------
    st.subheader("Model Comparison")
    compare_models(st.session_state.trained_models, st.session_state.X_test, st.session_state.y_test, st.session_state.model_info)

    # -------------------------------------------------
    # All Trained Models: Download Buttons
    # -------------------------------------------------
    st.subheader("All Trained Models")
    if st.session_state.trained_models:
        for i, (model_obj, label_str) in enumerate(st.session_state.trained_models):
            mo_index = i + 1
            # If it's a CV object, we can show best_estimator_. If not, show model_obj
            if hasattr(model_obj, "best_estimator_"):
                mo_name = model_obj.best_estimator_.__class__.__name__
            else:
                mo_name = model_obj.__class__.__name__

            st.markdown(f"**Model {mo_index}**: {mo_name} ({label_str})")

            # Create a unique key for the download button so each model can be downloaded
            dl_key = f"download_button_{mo_index}_{label_str}"

            # Pickle the model
            model_pickle = pickle.dumps(model_obj)
            st.download_button(
                label=f"Download Model {mo_index}",
                data=model_pickle,
                file_name=f"trained_model_{mo_name.lower()}_{label_str.lower()}.pkl",
                key=dl_key
            )
    else:
        st.write("No trained models yet. Train a model to see them here.")


if __name__ == '__main__':
    main()
