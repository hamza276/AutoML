import streamlit as st
import time
import optuna
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------------------
# Utility: GridSearchCV with fallback param grid
# ------------------------------------------------------------------------
def train_model_with_grid_search(model, param_grid, X_train, y_train):
    if not param_grid or all(len(v) == 0 for v in param_grid.values()):
        st.warning("No valid parameter grid found. Using minimal fallback param grid.")
        if hasattr(model, 'max_iter'):
            param_grid = {'max_iter': [100, 200]}
        else:
            param_grid = {'dummy_param': [True, False]}

    grid = GridSearchCV(model, param_grid, refit=True, cv=3, error_score='raise')
    with st.spinner("Training model (Grid Search)... Please wait"):
        time.sleep(1)
        try:
            grid.fit(X_train, y_train)
            st.success("Model Trained Successfully via Grid Search!")
            return grid
        except Exception as e:
            st.error(f"Failed to train model: {e}")

# ------------------------------------------------------------------------
# Utility: RandomizedSearchCV with fallback param distribution
# ------------------------------------------------------------------------
def train_model_with_random_search(model, param_dist, X_train, y_train):
    if not param_dist or all(len(v) == 0 for v in param_dist.values()):
        st.warning("No valid parameter distribution found. Using minimal fallback.")
        if hasattr(model, 'max_iter'):
            param_dist = {'max_iter': [100, 200, 500]}
        else:
            param_dist = {'dummy_param': [True, False]}

    rand_search = RandomizedSearchCV(model, param_dist, refit=True, cv=3, n_iter=5, error_score='raise')
    with st.spinner("Training model (Random Search)... Please wait"):
        time.sleep(1)
        try:
            rand_search.fit(X_train, y_train)
            st.success("Model Trained Successfully via Random Search!")
            return rand_search
        except Exception as e:
            st.error(f"Failed to train model: {e}")

# ------------------------------------------------------------------------
# Utility: Optuna
# ------------------------------------------------------------------------
def train_model_with_optuna(model_class, X_train, y_train, task, model_type):
    def objective(trial):
        if model_type in ['Logistic Regression', 'SVM Classifier']:
            c_value = trial.suggest_float("C", 0.01, 10.0, log=True)
            model_obj = model_class(C=c_value)
        elif model_type == 'Decision Tree Regressor':
            max_depth = trial.suggest_int("max_depth", 1, 20)
            model_obj = model_class(max_depth=max_depth)
        else:
            if hasattr(model_class(), 'max_depth'):
                max_depth = trial.suggest_int("max_depth", 1, 20)
                model_obj = model_class(max_depth=max_depth)
            else:
                model_obj = model_class()

        model_obj.fit(X_train, y_train)
        if task == 'Classification':
            preds = model_obj.predict(X_train)
            accuracy = (preds == y_train).mean()
            return accuracy
        else:
            preds = model_obj.predict(X_train)
            mse = mean_squared_error(y_train, preds)
            return -mse

    direction = 'maximize' if task == 'Classification' else 'minimize'
    study = optuna.create_study(direction=direction)
    with st.spinner("Tuning hyperparameters using Optuna... Please wait"):
        study.optimize(objective, n_trials=10, show_progress_bar=True)
    st.success("Optuna hyperparameter search completed!")

    best_params = study.best_params
    st.write("Best Optuna Hyperparameters:", best_params)

    if model_type in ['Logistic Regression', 'SVM Classifier']:
        if 'C' in best_params:
            final_model = model_class(C=best_params['C'])
        else:
            final_model = model_class()
    elif model_type == 'Decision Tree Regressor':
        if 'max_depth' in best_params:
            final_model = model_class(max_depth=best_params['max_depth'])
        else:
            final_model = model_class()
    else:
        if hasattr(model_class(), 'max_depth') and 'max_depth' in best_params:
            final_model = model_class(max_depth=best_params['max_depth'])
        else:
            final_model = model_class()

    final_model.fit(X_train, y_train)
    return final_model

def get_model_class(m_type):
     if m_type == 'Logistic Regression':
         return LogisticRegression
     elif m_type == 'Decision Tree Classifier':
         return DecisionTreeClassifier
     elif m_type == 'Random Forest Classifier':
         return RandomForestClassifier
     elif m_type == 'SVM Classifier':
         return SVC
     elif m_type == 'MLP Classifier':
         return MLPClassifier
     elif m_type == 'KNeighbors Classifier':
         return KNeighborsClassifier
     elif m_type == 'GaussianNB':
         return GaussianNB
     elif m_type == 'AdaBoost Classifier':
         return AdaBoostClassifier
     elif m_type == 'GradientBoosting Classifier':
         return GradientBoostingClassifier
     elif m_type == 'ExtraTrees Classifier':
         return ExtraTreesClassifier
     elif m_type == 'Decision Tree Regressor':
         return DecisionTreeRegressor
     elif m_type == 'Random Forest Regressor':
         return RandomForestRegressor
     elif m_type == 'SVM Regressor':
         return SVR
     elif m_type == 'MLP Regressor':
         return MLPRegressor
     elif m_type == 'Linear Regression':
         return LinearRegression
     elif m_type == 'Ridge Regression':
         return Ridge
     elif m_type == 'Lasso Regression':
         return Lasso
     elif m_type == 'ElasticNet Regression':
         return ElasticNet
     elif m_type == 'KNeighbors Regressor':
         return KNeighborsRegressor
     elif m_type == 'AdaBoost Regressor':
         return AdaBoostRegressor
     return None
