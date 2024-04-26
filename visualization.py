import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def run_visualizations(df):
    st.header('Visualization Page')

    # Visualizations code here
    st.subheader('Visualizations:')
    run_visualization = st.button('Run Visualization')

    if run_visualization:
        # Histogram of a numerical column
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_columns) > 0:
            selected_column = st.selectbox('Select a numeric column for histogram', numeric_columns)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_column], bins=20, kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.write('No numeric columns in the dataset for histogram.')

        # Pairplot for numeric columns
        if len(numeric_columns) > 1:
            st.write('Pairplot for numeric columns:')
            pairplot_fig = sns.pairplot(df[numeric_columns])
            st.pyplot(pairplot_fig)
        else:
            st.write('Insufficient numeric columns for pairplot.')

        # Correlation heatmap
        st.write('Correlation Heatmap:')
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            fig, ax = plt.subplots()
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)
        else:
            st.write('No numeric columns to compute correlation.')
