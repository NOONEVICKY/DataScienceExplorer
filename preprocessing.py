# Import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df, target_variable):
    # Split features and target variable
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle unknown categories in the test data
    encoder = OneHotEncoder(handle_unknown='ignore')  # You can also use 'error' or 'leave', depending on your strategy
    X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
    X_test_encoded = encoder.transform(X_test[categorical_columns])

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=['float64', 'int64']))
    X_test_scaled = scaler.transform(X_test.select_dtypes(include=['float64', 'int64']))

    # Combine encoded categorical features and scaled numerical features
    X_train_processed = pd.concat([pd.DataFrame(X_train_encoded.toarray()), pd.DataFrame(X_train_scaled)], axis=1)
    X_test_processed = pd.concat([pd.DataFrame(X_test_encoded.toarray()), pd.DataFrame(X_test_scaled)], axis=1)

    return X_train_processed, X_test_processed, y_train, y_test

def preprocessing_page(df):
    st.header('Preprocessing Page')
    st.write('Preprocess the dataset before model training.')

    target_variable = st.selectbox('Select the target variable', df.columns)

    preprocess_button = st.button('Preprocess Data')

    if preprocess_button:
        X_train, X_test, y_train, y_test = preprocess_data(df, target_variable)
        st.write('Data preprocessing completed.')

        st.subheader('Processed Data')
        st.write('Training Features:')
        st.dataframe(X_train.head())  # Display first few rows of processed training features
        st.write(f'Training Features Shape: {X_train.shape}')
        st.write('Training Target:')
        st.dataframe(y_train.head())  # Display first few rows of training target variable
        st.write('Testing Features:')
        st.dataframe(X_test.head())  # Display first few rows of processed testing features
        st.write(f'Testing Features Shape: {X_test.shape}')
        st.write('Testing Target:')
        st.dataframe(y_test.head())  # Display first few rows of testing target variable

# Main Streamlit app
def main():
    st.title('Your Data Science Dashboard')

    uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        preprocessing_page(df)  # Call the preprocessing page function

# Entry point of the app
if __name__ == '__main__':
    main()  
