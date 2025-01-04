import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------
# Utility: EDA
# ------------------------------------------------------------------------
def exploratory_data_analysis(df):
    st.write("Summary Statistics:", df.describe())
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
    st.pyplot(fig)
