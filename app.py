import streamlit as st
import pandas as pd
from model_select import ModelSelection
from preprocessing import preprocess_data
from feature_eng import feature_engineering
from visualization import run_visualizations
from data_analytics import data_analytics_page


# Initialize session state to store data between pages
if 'data' not in st.session_state:
    st.session_state.data = {}

# Title of the page
st.title('Data Science Dashboard')

# File uploader
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv', 'txt'])

# Check if file is uploaded and not empty
if uploaded_file is not None and uploaded_file.size > 0:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        if df.empty:
            st.warning("The uploaded file is empty.")
        else:
            st.session_state.data['df'] = df  # Store the uploaded dataframe in session state
    except Exception as e:
        st.error(f"Error reading the file: {e}")
else:
    st.write('Upload a non-empty CSV file to get started.')

# Sidebar navigation
navigation = st.sidebar.radio(
    'Navigate to',
    ['Home', 'Preprocessing', 'Feature Engineering', 'Visualization', 'Model', 'Data Analytics']
)

# Handle navigation based on user selection
if navigation == 'Home':
    st.write('Upload a CSV file to get started.')

elif navigation == 'Preprocessing':
    st.title('Preprocessing Page')
    st.write('Preprocess the dataset before model training.')

    target_variable = st.selectbox('Select the target variable', df.columns, key='target_variable')
    st.session_state.data['target_variable'] = target_variable

    preprocess_button = st.button('Preprocess Data')

    if preprocess_button:
        X_train_processed, X_test_processed, y_train, y_test = preprocess_data(df, target_variable)
        st.write('Data preprocessing completed.')
        st.write('X_train_processed:', X_train_processed.shape)
        st.write('X_test_processed:', X_test_processed.shape)

elif navigation == 'Feature Engineering':
    st.title('Feature Engineering Page')
    st.write('Apply feature engineering techniques to enhance model performance.')

    target_variable = st.session_state.data.get('target_variable')
    if target_variable is None:
        st.error("Please select the target variable first in the preprocessing step.")
    else:
        engineer_button = st.button('Apply Feature Engineering')

        if engineer_button:
            X_train_processed, X_test_processed, y_train, y_test = preprocess_data(df, target_variable)
            X_train_engineered, X_test_engineered = feature_engineering(X_train_processed, X_test_processed)
            st.write('Feature engineering completed.')
            st.write('X_train_engineered:', X_train_engineered.shape)
            st.write('X_test_engineered:', X_test_engineered.shape)

elif navigation == 'Visualization':
    st.title('Visualization Page')
    st.write('Visualize the data.')

    df_to_visualize = st.session_state.data.get('df')
    if df_to_visualize is not None:
        run_visualizations(df_to_visualize)
    else:
        st.warning("Please upload a CSV file and preprocess the data to visualize.")

elif navigation == 'Model':
    st.title('Model Page')
    st.write('Dataset Analysis and ML Operations.')

    target_variable = st.session_state.data.get('target_variable')
    if target_variable is None:
        st.error("Please select the target variable first in the preprocessing step.")
    else:
        run_model_button = st.button('Run Model')

        if run_model_button:
            model_selector = ModelSelection(df, target_variable)
            model_selector.preprocess_data()
            X_train_processed, X_test_processed, y_train, y_test = model_selector.get_preprocessed_data()
            model_selector.feature_engineering(X_train_processed, X_test_processed)
            mods = model_selector.train_and_test_models()



            # Display model results
            for model_name, results in mods.items():
                st.write(f'Model: {model_name}')
                st.write('Accuracy:', results['accuracy'])
                st.write('Confusion Matrix:')
                st.write(results['confusion_matrix'])
                st.write('Classification Report:')
                st.write(results['classification_report'])

elif navigation == 'Data Analytics':
    df_to_analyze = st.session_state.data.get('df')
    if df_to_analyze is not None:
        data_analytics_page(df_to_analyze)
    else:
        st.warning("Please upload a CSV file to perform data analytics.")