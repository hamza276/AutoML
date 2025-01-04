# AutoML Application with Multiple Fine-Tuning Methods

## Overview
This AutoML application, developed using **Streamlit**, provides an intuitive interface for users to perform various machine learning tasks, including classification and regression. The tool supports a wide range of machine learning models and fine-tuning methods, including manual hyperparameter selection, grid search, randomized search, and Optuna-based hyperparameter optimization.

## Features
- **File Upload Support:**
  - Supports CSV and Excel (XLSX) file formats.
- **Exploratory Data Analysis (EDA):**
  - Display dataset shape, columns, and basic statistics.
  - Generate correlation heatmaps.
- **Data Preprocessing:**
  - Handle missing values using different strategies (Mean, Median, Interpolation, or Drop).
  - Encode categorical variables for both classification and regression tasks.
  - Standardize features using `StandardScaler`.
- **Model Selection:**
  - Supports a wide variety of models for classification and regression.
  - Classification models:
    - Logistic Regression
    - Decision Tree Classifier
    - Random Forest Classifier
    - SVM Classifier
    - MLP Classifier
    - KNeighbors Classifier
    - Gaussian Naive Bayes
    - AdaBoost Classifier
    - GradientBoosting Classifier
    - ExtraTrees Classifier
  - Regression models:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - ElasticNet Regression
    - Decision Tree Regressor
    - Random Forest Regressor
    - SVM Regressor
    - MLP Regressor
    - KNeighbors Regressor
    - AdaBoost Regressor
    - GradientBoosting Regressor
    - ExtraTrees Regressor

- **Hyperparameter Tuning:**
  - Manual hyperparameter tuning via sliders and dropdowns.
  - Automated fine-tuning using:
    - **Grid Search**
    - **Randomized Search**
    - **Optuna Optimization**

- **Model Training and Testing:**
  - Train models with user-defined hyperparameters or automated fine-tuning.
  - Test trained models and display performance metrics.
    - Classification metrics: Accuracy, Precision, Recall, F1-Score.
    - Regression metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).

- **Model Comparison:**
  - Compare multiple trained models based on selected metrics.
  - Visualize model comparison results using bar charts.

- **Model Download:**
  - Download trained models in pickle format for future use.
## Project Structure
automl_app/
├── __init__.py
├── app.py         # Main Streamlit app
├── utils/
│   ├── __init__.py
│   ├── data_handler.py  # Data loading and preprocessing
│   ├── eda.py         # Exploratory data analysis
│   ├── model_trainer.py # Model training functions
│   └── model_evaluation.py # Model evaluation and comparison


## How to Use
1. **Upload Dataset:**
   - Select the file type (CSV or Excel) and upload your dataset.
2. **Select Preferences:**
   - Choose whether to enable EDA and descriptive statistics.
   - Select the task type (Classification or Regression).
   - Choose the missing value handling strategy.
3. **Select Model and Hyperparameter Tuning:**
   - Choose a model from the provided list.
   - Select the hyperparameter tuning approach:
     - No Tuning (Default hyperparameters)
     - Manual Hyperparams (Set values manually)
     - Complete Fine-Tuning (Grid Search, Randomized Search, or Optuna)
4. **Train the Model:**
   - Click the "Train Model" button to start the training process.
5. **Test the Model:**
   - Once training is complete, click the "Test Model" button to evaluate the model.
6. **Compare Models:**
   - Select trained models for comparison and visualize the results.
7. **Download Trained Models:**
   - Download individual models or the best-performing model in pickle format.

## Requirements
- **Python 3.7+**
- Required Python libraries:
  ```bash
  pip install streamlit pandas numpy scikit-learn optuna matplotlib seaborn
  ```

## Installation and Execution
1. Clone the repository or download the script.
2. Install the required libraries using the provided command.
3. Run the application using Streamlit:
   ```bash
   python -m streamlit run app.py
   ```
4. Access the application via the provided localhost link in your web browser.

## Example Usage
1. Upload a CSV file containing a dataset.
2. Select "Classification" as the task type.
3. Choose a classification model (e.g., Random Forest Classifier).
4. Enable EDA and choose "Mean" as the missing value handling strategy.
5. Select "Complete Fine-Tuning" and choose "Grid Search" for hyperparameter optimization.
6. Train the model and download the best-performing model.

## Additional Information
- This application uses **Optuna** for advanced hyperparameter optimization.
- Fine-tuned models are saved in session and can be downloaded individually.
- The comparison feature allows users to identify the best-performing model based on selected metrics.

## Limitations
- The current version supports only CSV and Excel file formats.
- The number of trials for Optuna-based optimization is set to 10 by default.
- The application requires the dataset to be clean and properly formatted.

## License
This project is licensed under the MIT License.

## Acknowledgments
- The application leverages popular machine learning libraries such as **scikit-learn** and **Optuna**.
- Special thanks to the Streamlit community for providing an excellent framework for building data applications.

---
For any issues or feature requests, please contact hafizhamzakhan1997@gmail.com.

