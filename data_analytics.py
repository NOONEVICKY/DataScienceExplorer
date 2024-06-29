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
from io import StringIO , BytesIO
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


def data_analytics_page(df):

    def get_dataframe_info(df):
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        return s

    def data_summary_page(df):
        st.header('Data Summary')

        if df.empty:
            st.write("The dataset is empty. Please upload a valid dataset.")
        else:
            pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
            dataframe_info = get_dataframe_info(df)
            

            # Use applymap for element-wise operation and check data type before applying format
            descriptive_stats = df.describe(include='all').applymap(lambda x: format(x, 'g') if isinstance(x, (int, float)) else x)

            st.dataframe(descriptive_stats)
    data_summary_page(df)
    
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
    try:
        missing_values = df.isnull().sum()
        if not missing_values.any():
            st.write('No missing values found.')
        else:
            st.write('Missing values detected:')
            st.dataframe(missing_values[missing_values > 0])
    
            # Get list of numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
            for col in df.columns:
                if df[col].isnull().any():
                    if col in numeric_cols:
                        fill_strategy = st.selectbox(f"Choose a strategy to fill missing values for numeric column '{col}'", 
                                                     ["Constant", "Most Frequent"], key=f'{col}_fill_strategy')
                    else:
                        fill_strategy = st.selectbox(f"Choose a strategy to fill missing values for categorical column '{col}'", 
                                                     ["Constant", "Most Frequent"], key=f'{col}_fill_strategy')
    
                    if fill_strategy == "Constant":
                        if col in numeric_cols:
                            const_value = st.number_input(f"Enter the constant value for numeric column '{col}':", key=f'const_{col}')
                        else:
                            const_value = st.text_input(f"Enter the constant value for categorical column '{col}':", key=f'const_{col}')
                        imputer = SimpleImputer(strategy='constant', fill_value=const_value)
                    elif fill_strategy == "Most Frequent":
                        imputer = SimpleImputer(strategy='most_frequent')
    
                    df[col] = imputer.fit_transform(df[[col]]).ravel()
    
            st.write('Missing values have been handled. Updated dataset:')
            st.dataframe(df.head())  # Display the top rows of the DataFrame after handling missing values
    except Exception as e:
        st.error(f"Error handling missing values: {e}")


    # Outlier Detection and Treatment
    st.subheader('Outlier Detection and Treatment')
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        selected_column = st.selectbox('Select column for outlier detection', numeric_columns)
        if st.button('Detect and Remove Outliers'):
            isolation_forest = IsolationForest(contamination='auto')
            df['Outlier'] = isolation_forest.fit_predict(df[[selected_column]])
            outliers_count = df[df['Outlier'] == -1].shape[0]
            if outliers_count > 0:
                df.drop(index=df[df['Outlier'] == -1].index, inplace=True)
                st.write(f'Outliers detected and removed: {outliers_count}')
            else:
                st.write('No outliers detected.')
            df.drop(columns=['Outlier'], inplace=True)
            st.write(df.head())

    # Handling Categorical Variables
    st.subheader('Handling Categorical Variables')
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        selected_col = st.selectbox('Select a categorical column', categorical_columns)
        encoding_type = st.selectbox('Select encoding type', ['Label Encoding', 'One-Hot Encoding'])
        if encoding_type == 'Label Encoding':
            encoder = LabelEncoder()
            df[selected_col + '_label_encoded'] = encoder.fit_transform(df[selected_col])
        elif encoding_type == 'One-Hot Encoding':
            encoder = OneHotEncoder(sparse_output=False)
            encoded_data = encoder.fit_transform(df[[selected_col]])
            df = pd.concat([df.drop(columns=[selected_col]), pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([selected_col]))], axis=1)
        st.write(df.head())
    st.markdown("---")

    # Scaling/Normalizing Numerical Features
    st.subheader('Scaling/Normalizing Numerical Features')
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.write("No numeric columns available for scaling.")
        return
    selected_num_col = st.selectbox('Select a numeric column for scaling/normalization', numeric_columns)
    scaling_type = st.selectbox('Select scaling/normalization type', ['StandardScaler', 'MinMaxScaler'])
    try:
        scaler = StandardScaler() if scaling_type == 'StandardScaler' else MinMaxScaler()
        df[selected_num_col + '_scaled'] = scaler.fit_transform(df[[selected_num_col]])
        st.write(df[[selected_num_col, selected_num_col + '_scaled']])
    except Exception as e:
        st.error(f"Error scaling {selected_num_col}: {e}")

    st.markdown("---")
    
    # Display updated DataFrame after handling missing values
    st.subheader('Updated DataFrame after Handling Missing Values')
    st.write(df.head())

    st.markdown("---")

    # Model Evaluation and Comparison
    st.subheader('Model Evaluation and Comparison')

    # Ensure DataFrame is suitable for modeling
    if df.empty:
        st.write("The dataset is empty. Cannot perform model evaluation.")
        return

    # Ensure there are numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        st.write("No numeric columns available for model evaluation.")
        return

    # Preprocess categorical variables using one-hot encoding
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Dynamic selection of features and target variable
    all_columns = df.columns.tolist()
    target_column = st.selectbox('Select target variable:', numeric_columns)
    features = [col for col in all_columns if col != target_column]

    if not features:
        st.write("No features available for modeling.")
        return

    # Data splitting
    try:
        X = df[features]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        st.error(f"Error in data splitting: {e}")
        return

    # Model training and evaluation
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Decision Tree Regressor': DecisionTreeRegressor(),
        'Random Forest Regressor': RandomForestRegressor()
    }

    results = []
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results.append({'Model': name, 'MSE': mse, 'MAE': mae, 'R²': r2})
        except Exception as e:
            st.write(f"Error training {name}: {e}")

    if results:
        results_df = pd.DataFrame(results)
        st.table(results_df.style.highlight_min(subset=['MSE', 'MAE'], color='lightgreen').highlight_max(subset=['R²'], color='lightgreen'))

        # Additional visualizations for residuals if models were successfully trained
        if 'Linear Regression' in models:
            if st.checkbox('Show Residual Plots'):
                for name, model in models.items():
                    try:
                        y_pred = model.predict(X_test)
                        fig, ax = plt.subplots()
                        sns.residplot(x=y_test, y=y_pred, lowess=True, color="g", ax=ax)
                        ax.set_title(f'Residual Plot for {name}')
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error generating residual plot for {name}: {e}")
    
    
    
    
    # Time Series Analysis
    st.subheader('Time Series Analysis')

    # Check if there are datetime columns available
    time_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    if not time_columns:
        st.write("No datetime columns available for time series analysis.")
    else:
        time_col = st.selectbox('Select Time Series Column:', time_columns)
        period = st.number_input('Define the Periodicity:', min_value=1, max_value=365, step=1)
        
        if st.button('Decompose Time Series'):
            result = seasonal_decompose(df[time_col], model='additive', period=period)
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
            result.trend.plot(ax=ax1)
            result.seasonal.plot(ax=ax2)
            result.resid.plot(ax=ax3)
            result.observed.plot(ax=ax4)
            st.pyplot(fig)

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
    st.subheader('Statistical Analysis')

    # T-Test
    st.write("### T-Test")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            numeric_col1 = st.selectbox('Select first numeric column for t-test', numeric_cols)
        with col2:
            numeric_col2 = st.selectbox('Select second numeric column for t-test', numeric_cols, index=min(1, len(numeric_cols)-1))

        if st.button('Perform t-test'):
            t_statistic, p_value = ttest_ind(df[numeric_col1].dropna(), df[numeric_col2].dropna())
            st.write(f'T-statistic: {t_statistic:.4f}')
            st.write(f'P-value: {p_value:.4f}')
    else:
        st.write("Not enough numeric columns for t-test")

    st.markdown("---")

    # ANOVA
    st.write("### ANOVA")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
        numeric_col = st.selectbox('Select numeric column for ANOVA', numeric_cols)
        categorical_col = st.selectbox('Select categorical column for ANOVA', categorical_cols)

        if st.button('Perform ANOVA'):
            groups = [group for name, group in df.groupby(categorical_col)[numeric_col] if len(group) > 0]
            if len(groups) >= 2:
                f_statistic, p_value = f_oneway(*groups)
                st.write(f'F-statistic: {f_statistic:.4f}')
                st.write(f'P-value: {p_value:.4f}')
            else:
                st.write('Not enough valid groups to perform ANOVA.')
    else:
        st.write("Not enough appropriate columns for ANOVA")

    st.markdown("---")

    # Correlation
    st.write("### Correlation Analysis")
    if len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            numeric_col1 = st.selectbox('Select first numeric column for correlation', numeric_cols)
        with col2:
            numeric_col2 = st.selectbox('Select second numeric column for correlation', numeric_cols, index=min(1, len(numeric_cols)-1))

        if st.button('Calculate Correlation'):
            corr_coef, p_value = pearsonr(df[numeric_col1].dropna(), df[numeric_col2].dropna())
            st.write(f'Correlation Coefficient: {corr_coef:.4f}')
            st.write(f'P-value: {p_value:.4f}')
    else:
        st.write("Not enough numeric columns for correlation analysis")

    st.markdown("---")

    # One-sample proportion test
    st.write("### One-Sample Proportion Test")
    if len(categorical_cols) > 0:
        categorical_col = st.selectbox('Select categorical column for proportion test', categorical_cols)
        if categorical_col:
            unique_values = df[categorical_col].unique()
            success_value = st.selectbox('Select the success value', unique_values)
            proportion = st.number_input('Enter the hypothesized proportion (between 0 and 1)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

            if st.button('Perform Proportion Test'):
                count = df[categorical_col].eq(success_value).sum()
                total = df[categorical_col].count()
                observed_proportion = count / total
                p_value = 1 - binom.cdf(count - 1, total, proportion)
                st.write(f'Observed Proportion: {observed_proportion:.4f}')
                st.write(f'P-value: {p_value:.4f}')
    else:
        st.write("No categorical columns available for proportion test")

    st.markdown("---")

    # Chi-square test
    st.write("### Chi-Square Test")
    if len(categorical_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            categorical_col1 = st.selectbox('Select first categorical column for chi-square test', categorical_cols)
        with col2:
            categorical_col2 = st.selectbox('Select second categorical column for chi-square test', categorical_cols, index=min(1, len(categorical_cols)-1))

        if st.button('Perform Chi-square Test'):
            contingency_table = pd.crosstab(df[categorical_col1], df[categorical_col2])
            chi2_statistic, p_value, dof, expected = chi2_contingency(contingency_table)
            st.write(f'Chi-square statistic: {chi2_statistic:.4f}')
            st.write(f'P-value: {p_value:.4f}')
    else:
        st.write("Not enough categorical columns for chi-square test")


    st.subheader('Correlation Matrix and Data Distribution')
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        corr_matrix = df[numeric_columns].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        st.markdown("---")

        selected_column = st.selectbox('Select Column for Distribution Plot:', numeric_columns)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_column], kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numeric columns available for correlation or distribution plots.")
        

        # Add this at the end of your data_analytics_page function
    st.markdown("---")
    st.subheader("Generate PDF Report")
    if st.button("Generate PDF Report"):
        pdf = create_pdf_report(df)
        st.download_button(
            label="Download PDF Report",
            data=pdf,
            file_name="data_analytics_report.pdf",
            mime="application/pdf"
        )

def create_pdf_report(df, summary_stats, missing_values_summary, outlier_summary, categorical_summary, scaling_summary, model_performance, time_series_analysis, statistical_results, updated_df_head):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("Data Analytics Report", styles['Title']))
    elements.append(Spacer(1, 12))

    # 1. Dataset Overview
    elements.append(Paragraph("1. Dataset Overview", styles['Heading2']))
    overview_data = [
        ["Number of rows", str(df.shape[0])],
        ["Number of columns", str(df.shape[1])],
        ["Column names and data types", str(df.dtypes.to_dict())]
    ]
    t = Table(overview_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    # 2. Summary Statistics
    elements.append(Paragraph("2. Summary Statistics", styles['Heading2']))
    summary_stats_data = [summary_stats.columns.tolist()] + summary_stats.values.tolist()
    t = Table(summary_stats_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    # 3. Missing Values
    elements.append(Paragraph("3. Missing Values", styles['Heading2']))
    missing_values_data = [["Column", "Missing Values"]] + missing_values_summary
    t = Table(missing_values_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    # 4. Outlier Detection and Treatment
    elements.append(Paragraph("4. Outlier Detection and Treatment", styles['Heading2']))
    elements.append(Paragraph("Summary of outliers detected:", styles['Normal']))
    elements.append(Paragraph(outlier_summary, styles['Normal']))
    elements.append(Spacer(1, 12))

    # 5. Handling Categorical Variables
    elements.append(Paragraph("5. Handling Categorical Variables", styles['Heading2']))
    elements.append(Paragraph("Summary of encoding methods used:", styles['Normal']))
    elements.append(Paragraph(categorical_summary, styles['Normal']))
    elements.append(Spacer(1, 12))

    # 6. Scaling/Normalizing Numerical Features
    elements.append(Paragraph("6. Scaling/Normalizing Numerical Features", styles['Heading2']))
    elements.append(Paragraph("Summary of scaling/normalization methods used:", styles['Normal']))
    elements.append(Paragraph(scaling_summary, styles['Normal']))
    elements.append(Spacer(1, 12))

    # 7. Model Evaluation and Comparison
    elements.append(Paragraph("7. Model Evaluation and Comparison", styles['Heading2']))
    elements.append(Paragraph("List of models evaluated:", styles['Normal']))
    elements.append(Paragraph(model_performance, styles['Normal']))
    elements.append(Spacer(1, 12))

    # 8. Time Series Analysis
    elements.append(Paragraph("8. Time Series Analysis", styles['Heading2']))
    elements.append(Paragraph("Decomposition of time series data:", styles['Normal']))
    elements.append(Paragraph(time_series_analysis, styles['Normal']))
    elements.append(Spacer(1, 12))

    # 9. Statistical Analysis Results
    elements.append(Paragraph("9. Statistical Analysis Results", styles['Heading2']))
    elements.append(Paragraph(statistical_results, styles['Normal']))
    elements.append(Spacer(1, 12))

    # 10. Updated DataFrame
    elements.append(Paragraph("10. Updated DataFrame", styles['Heading2']))
    elements.append(Paragraph("Display of the updated DataFrame after handling missing values and outliers:", styles['Normal']))
    updated_df_data = [updated_df_head.columns.tolist()] + updated_df_head.values.tolist()
    t = Table(updated_df_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 12))

    doc.build(elements)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

# Example data and usage
# Assuming you have a DataFrame df with your data
data = {
    'Column1': [1, 2, 3, 4, 5],
    'Column2': [10, 20, 30, 40, 50],
    'Column3': ['A', 'B', 'C', 'D', 'E']
}
df = pd.DataFrame(data)

# Example summary data
summary_stats = df.describe()
missing_values_summary = [(col, df[col].isnull().sum()) for col in df.columns if df[col].isnull().sum() > 0]
outlier_summary = "Outlier detection summary text here."
categorical_summary = "Categorical encoding summary text here."
scaling_summary = "Scaling/normalization methods summary text here."
model_performance = "Model performance metrics summary text here."
time_series_analysis = "Time series decomposition summary text here."
statistical_results = "Statistical analysis results summary text here."
updated_df_head = df.head()

# Generate the PDF report
pdf_data = create_pdf_report(df, summary_stats, missing_values_summary, outlier_summary, categorical_summary, scaling_summary, model_performance, time_series_analysis, statistical_results, updated_df_head)

# Save the PDF
with open('data_analytics_report.pdf', 'wb') as f:
    f.write(pdf_data)