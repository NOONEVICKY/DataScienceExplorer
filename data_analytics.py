import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, pearsonr, chi2_contingency
from scipy.stats.distributions import binom
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def data_analytics_page(df):
    st.header('Data Analytics Page')

    # Data summary
    st.subheader('Data Summary')
    st.write(df.info())
    st.write(df.describe(include='all'))
    st.markdown("---")
    
    # Exploratory Data Analysis (EDA)
    st.subheader('Exploratory Data Analysis (EDA)')
    plot_types = ['Scatter Plot', 'Histogram', 'Box Plot', 'Violin Plot']
    selected_plot = st.selectbox('Select plot type', plot_types)

    if selected_plot == 'Scatter Plot':
        st.write('Select two numeric columns for scatter plot analysis')
        scatter_col1 = st.selectbox('Select X-axis column', df.select_dtypes(include=['float64', 'int64']).columns)
        scatter_col2 = st.selectbox('Select Y-axis column', df.select_dtypes(include=['float64', 'int64']).columns)

        if st.button('Generate Scatter Plot'):
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=scatter_col1, y=scatter_col2, ax=ax)
            ax.set_xlabel(scatter_col1)
            ax.set_ylabel(scatter_col2)
            st.pyplot(fig)

    elif selected_plot == 'Histogram':
        st.write('Select a numeric column for histogram')
        hist_col = st.selectbox('Select column for histogram', df.select_dtypes(include=['float64', 'int64']).columns)

        if st.button('Generate Histogram'):
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=hist_col, kde=True, ax=ax)
            ax.set_xlabel(hist_col)
            ax.set_ylabel('Count')
            st.pyplot(fig)

    elif selected_plot == 'Box Plot':
        st.write('Select a numeric column for box plot')
        box_col = st.selectbox('Select column for box plot', df.select_dtypes(include=['float64', 'int64']).columns)

        if st.button('Generate Box Plot'):
            fig, ax = plt.subplots()
            sns.boxplot(data=df, y=box_col, ax=ax)
            ax.set_ylabel(box_col)
            st.pyplot(fig)

    elif selected_plot == 'Violin Plot':
        st.write('Select a numeric column for violin plot')
        violin_col = st.selectbox('Select column for violin plot', df.select_dtypes(include=['float64', 'int64']).columns)

        if st.button('Generate Violin Plot'):
            fig, ax = plt.subplots()
            sns.violinplot(data=df, y=violin_col, ax=ax)
            ax.set_ylabel(violin_col)
            st.pyplot(fig)

    st.markdown("---")

    # Missing values
    st.subheader('Missing Values')
    missing_values = df.isnull().sum()
    if missing_values.empty:
        st.write('No missing values found.')
    else:
        st.write(missing_values)

    # Outlier Detection and Treatment
    st.subheader('Outlier Detection and Treatment')
    outlier_col = st.selectbox('Select a numeric column for outlier detection', df.select_dtypes(include=['float64', 'int64']).columns)
    if st.button('Detect Outliers'):
        isolation_forest = IsolationForest(contamination='auto')
        outlier_pred = isolation_forest.fit_predict(df[[outlier_col]])
        df['Outlier'] = outlier_pred
        st.write(df[df['Outlier'] == -1])  # Show detected outliers
        df = df[df['Outlier'] == 1].drop(columns=['Outlier'])  # Remove outliers from the DataFrame
    st.markdown("---")


    # Handling Categorical Variables
    st.header('Feature Engineering')

    # Handling Categorical Variables
    st.subheader('Handling Categorical Variables')
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        selected_col = st.selectbox('Select a categorical column for encoding', categorical_cols)
        encoding_type = st.selectbox('Select encoding type', ['Label Encoding', 'One-Hot Encoding'])

        if encoding_type == 'Label Encoding':
            # Perform label encoding
            label_encoder = LabelEncoder()
            df[selected_col + '_label_encoded'] = label_encoder.fit_transform(df[selected_col])
            st.write(df[[selected_col, selected_col + '_label_encoded']])
            st.write(f'Label encoded column name: {selected_col}_label_encoded')

        elif encoding_type == 'One-Hot Encoding':
            # Perform one-hot encoding
            label_encoder = LabelEncoder()
            df[selected_col + '_label_encoded'] = label_encoder.fit_transform(df[selected_col])
            onehot_encoder = OneHotEncoder(drop='first')
            encoded_features = onehot_encoder.fit_transform(df[[selected_col + '_label_encoded']])
            encoded_df = pd.DataFrame(encoded_features.toarray(), columns=[f'{selected_col}_{val}' for val in onehot_encoder.categories_[0][1:]])
            df = pd.concat([df, encoded_df], axis=1)
            st.write(df[[selected_col] + list(encoded_df.columns)])
            st.write('One-hot encoded columns:', list(encoded_df.columns))

    st.markdown("---")

    # Scaling/Normalizing Numerical Features
    st.header('Scaling/Normalizing Numerical Features')
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if numeric_cols:
        selected_num_col = st.selectbox('Select a numeric column for scaling/normalization', numeric_cols)
        scaling_type = st.selectbox('Select scaling/normalization type', ['StandardScaler', 'MinMaxScaler'])

        if scaling_type == 'StandardScaler':
            # Perform StandardScaler
            scaler = StandardScaler()
            df[selected_num_col + '_standard_scaled'] = scaler.fit_transform(df[[selected_num_col]])
            st.write(df[[selected_num_col, selected_num_col + '_standard_scaled']])
            st.write(f'Standard scaled column name: {selected_num_col}_standard_scaled')

        elif scaling_type == 'MinMaxScaler':
            # Perform MinMaxScaler
            scaler = MinMaxScaler()
            df[selected_num_col + '_minmax_scaled'] = scaler.fit_transform(df[[selected_num_col]])
            st.write(df[[selected_num_col, selected_num_col + '_minmax_scaled']])
            st.write(f'MinMax scaled column name: {selected_num_col}_minmax_scaled')

    st.markdown("---")

    # Missing values
    st.subheader('Missing Values')
    missing_values = df.isnull().sum()
    if missing_values.empty:
        st.write('No missing values found.')
    else:
        st.write(missing_values)

        # Handle missing numeric values
        missing_numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns[df.select_dtypes(include=['float64', 'int64']).isnull().any()].tolist()
        if missing_numeric_cols:
            imputer_numeric = SimpleImputer(strategy='mean')
            df_numeric = pd.DataFrame(imputer_numeric.fit_transform(df[missing_numeric_cols]), columns=missing_numeric_cols)
            df[missing_numeric_cols] = df_numeric

        # Handle missing categorical values
        missing_categorical_cols = df.select_dtypes(include=['object']).columns[df.select_dtypes(include=['object']).isnull().any()].tolist()
        if missing_categorical_cols:
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            df_categorical = pd.DataFrame(imputer_categorical.fit_transform(df[missing_categorical_cols]), columns=missing_categorical_cols)
            df[missing_categorical_cols] = df_categorical

        st.write('Missing values imputed successfully.')
    
    st.markdown("---")

    # Display updated DataFrame after handling missing values
    st.subheader('Updated DataFrame after Handling Missing Values')
    st.write(df.head())

    st.markdown("---")

    # Display updated DataFrame after feature engineering
    st.subheader('Updated DataFrame after Feature Engineering')
    st.write(df.head())

    st.markdown("---")

    st.subheader('Model Evaluation and Comparison')
    # Assuming df is your DataFrame containing the data
    X = df.select_dtypes(include=['float64', 'int64']).drop(columns=[outlier_col])  # Assuming outlier_col is excluded from features
    y = df[outlier_col]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing pipeline
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test = scaler.transform(imputer.transform(X_test))

    # List of models to evaluate
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Decision Tree Regressor': DecisionTreeRegressor(),
        'Random Forest Regressor': RandomForestRegressor()
    }

    # Dictionary to store evaluation results
    evaluation_results = {}

    # Iterate over each model
    for name, model in models.items():
        model.fit(X_train, y_train)  # Fit the model
        y_pred = model.predict(X_test)  # Make predictions

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store results in the evaluation dictionary
        evaluation_results[name] = {'MSE': mse, 'MAE': mae, 'R2 Score': r2}

    # Display the evaluation results
    st.subheader('Model Evaluation Results')
    for name, metrics in evaluation_results.items():
        st.write(f'{name}:')
        st.write(f'Mean Squared Error (MSE): {metrics["MSE"]:.2f}')
        st.write(f'Mean Absolute Error (MAE): {metrics["MAE"]:.2f}')
        st.write(f'R-squared (R2 Score): {metrics["R2 Score"]:.2f}')
        st.markdown("---")



    # Time Series Analysis
    st.subheader('Time Series Analysis')
    time_series_cols = df.select_dtypes(include=['datetime64']).columns.tolist()  # Get all datetime columns
    if len(time_series_cols) > 0:
        time_series_col = st.selectbox('Select a column for time series analysis', time_series_cols)
        if st.button('Perform Time Series Analysis'):
            decomposition = seasonal_decompose(df[time_series_col], model='additive', period=12)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
            decomposition.trend.plot(ax=ax1)
            decomposition.seasonal.plot(ax=ax2)
            decomposition.resid.plot(ax=ax3)
            decomposition.observed.plot(ax=ax4)
            ax1.set_ylabel('Trend')
            ax2.set_ylabel('Seasonal')
            ax3.set_ylabel('Residual')
            ax4.set_ylabel('Observed')
            st.pyplot(fig)
    else:
        st.write('No datetime columns found for time series analysis.')


    # Display updated DataFrame after outlier removal
    st.subheader('Updated DataFrame after Outlier Removal')
    st.write(df.head())
    st.markdown("---")

    # Correlation matrix
    st.subheader('Correlation Matrix')
    numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.write('No numeric columns to compute correlation.')
    st.markdown("---")
        

    # Data distribution
    st.subheader('Data Distribution')
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    selected_col = st.selectbox('Select a numeric column', numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(data=df, x=selected_col, kde=True, ax=ax)
    st.pyplot(fig)
    st.markdown("---")

    # Statistical analysis
    st.subheader('Statistical Analysis')
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns


    # t-test
    if len(numeric_cols) > 1:
        col1, col2 = st.columns(2)
        with col1:
            numeric_col1 = st.selectbox('Select a numeric column for t-test', numeric_cols)
        with col2:
            numeric_col2 = st.selectbox('Select another numeric column for t-test', numeric_cols, index=1)

        if st.button('Perform t-test'):
            t_statistic, p_value = ttest_ind(df[numeric_col1], df[numeric_col2])
            st.write(f'T-statistic: {t_statistic:.2f}')
            st.write(f'P-value: {p_value:.4f}')

            if p_value < 0.05:
                st.write('The p-value is less than 0.05, which suggests that the difference between the two groups is statistically significant. Therefore, we can reject the null hypothesis that the means of the two groups are equal.')
            else:
                st.write('The p-value is greater than or equal to 0.05, which suggests that the difference between the two groups is not statistically significant. Therefore, we fail to reject the null hypothesis that the means of the two groups are equal.')
    
    st.markdown("---")
    # ANOVA
    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
        numeric_col = st.selectbox('Select a numeric column for ANOVA', numeric_cols)
        categorical_col = st.selectbox('Select a categorical column for ANOVA', categorical_cols)

        if st.button('Perform ANOVA'):
            groups = df.groupby(categorical_col)[numeric_col].apply(list)
            f_statistic, p_value = f_oneway(*groups)
            st.write(f'F-statistic: {f_statistic:.2f}')
            st.write(f'P-value: {p_value:.4f}')

            if p_value < 0.05:
                st.write('The p-value is less than 0.05, which suggests that at least one group mean is significantly different from the others. Therefore, we can reject the null hypothesis that all group means are equal.')
            else:
                st.write('The p-value is greater than or equal to 0.05, which suggests that there is no significant difference between the group means. Therefore, we fail to reject the null hypothesis that all group means are equal.')

    st.markdown("---")
    # Correlation
    if len(numeric_cols) > 1:
        col1, col2 = st.columns(2)
        with col1:
            numeric_col1 = st.selectbox('Select a numeric column for correlation', numeric_cols)
        with col2:
            numeric_col2 = st.selectbox('Select another numeric column for correlation', numeric_cols, index=1)

        if st.button('Calculate Correlation'):
            corr_coef, p_value = pearsonr(df[numeric_col1], df[numeric_col2])
            st.write(f'Correlation Coefficient: {corr_coef:.2f}')
            st.write(f'P-value: {p_value:.4f}')

            if p_value < 0.05:
                st.write('The p-value is less than 0.05, which suggests that the correlation between the two variables is statistically significant. Therefore, we can reject the null hypothesis that there is no correlation between the variables.')
            else:
                st.write('The p-value is greater than or equal to 0.05, which suggests that the correlation between the two variables is not statistically significant. Therefore, we fail to reject the null hypothesis that there is no correlation between the variables.')

    st.markdown("---")
    # One-sample proportion test
    if len(categorical_cols) > 0:
        categorical_col = st.selectbox('Select a categorical column for proportion test', categorical_cols)
        if categorical_col:
            unique_values = df[categorical_col].unique()
            success_value = st.selectbox('Select the success value', unique_values)
            proportion = st.number_input('Enter the hypothesized proportion (between 0 and 1)', value=0.5, step=0.01)

            if st.button('Perform Proportion Test'):
                count = df[categorical_col].eq(success_value).sum()
                total = df[categorical_col].count()
                observed_proportion = count / total
                p_value = 1 - binom.cdf(count - 1, total, proportion)  # Calculate p-value using binom.cdf
                st.write(f'Observed Proportion: {observed_proportion:.2f}')
                st.write(f'P-value: {p_value:.4f}')

                if p_value < 0.05:
                    st.write(f'The p-value is less than 0.05, which suggests that the observed proportion ({observed_proportion:.2f}) is significantly different from the hypothesized proportion ({proportion:.2f}). Therefore, we can reject the null hypothesis that the population proportion is equal to the hypothesized proportion.')
                else:
                    st.write(f'The p-value is greater than or equal to 0.05, which suggests that the observed proportion ({observed_proportion:.2f}) is not significantly different from the hypothesized proportion ({proportion:.2f}). Therefore, we fail to reject the null hypothesis that the population proportion is equal to the hypothesized proportion.')

    st.markdown("---")
    # Chi-square test
    if len(categorical_cols) > 1:
        col1, col2 = st.columns(2)
        with col1:
            categorical_col1 = st.selectbox('Select a categorical column for chi-square test', categorical_cols)
        with col2:
            categorical_col2 = st.selectbox('Select another categorical column for chi-square test', categorical_cols, index=1)

        if st.button('Perform Chi-square Test'):
            contingency_table = pd.crosstab(df[categorical_col1], df[categorical_col2])
            chi2_statistic, p_value, dof, expected_frequencies = chi2_contingency(contingency_table)
            st.write(f'Chi-square statistic: {chi2_statistic:.2f}')
            st.write(f'P-value: {p_value:.4f}')

            if p_value < 0.05:
                st.write('The p-value is less than 0.05, which suggests that there is a significant association between the two categorical variables. Therefore, we can reject the null hypothesis that the variables are independent.')
            else:
                st.write('The p-value is greater than or equal to 0.05, which suggests that there is no significant association between the two categorical variables. Therefore, we fail to reject the null hypothesis that the variables are independent.')