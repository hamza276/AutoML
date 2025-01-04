import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# ------------------------------------------------------------------------
# Utility: Display dataset
# ------------------------------------------------------------------------
def display_dataset(df):
    st.write("Dataset Shape:", df.shape)
    st.write("Number of Columns:", len(df.columns))
    st.write("Column Names:", df.columns.tolist())
    st.write("Dataset Head:", df.head())

# ------------------------------------------------------------------------
# Utility: Handle missing values
# ------------------------------------------------------------------------
def handle_missing_values(df, strategy):
    if strategy == 'Mean':
        return df.fillna(df.mean())
    elif strategy == 'Median':
        return df.fillna(df.median())
    elif strategy == 'Interpolation':
        return df.interpolate()
    return df.dropna()

# ------------------------------------------------------------------------
# Utility: Split data
# ------------------------------------------------------------------------
def split_data(df, target_variable):
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]
    return X, y

# ------------------------------------------------------------------------
# Utility: Handle categorical variables
# ------------------------------------------------------------------------
def handle_categorical_variables(X, y, task):
    categorical_cols = X.select_dtypes(include=['object']).columns
    if task == 'Classification':
        encoder = OneHotEncoder(handle_unknown='ignore')
        X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]).toarray())
        X = pd.concat([X.drop(categorical_cols, axis=1), X_encoded], axis=1)
    else:  # Regression
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        for col in categorical_cols:
            X[col] = pd.Categorical(X[col]).codes
    return X, y
