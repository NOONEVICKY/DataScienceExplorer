Data Science Explorer App
This repository hosts a Streamlit application designed to serve as a comprehensive tool for data scientists. The app enables users to upload datasets, conduct exploratory data analysis (EDA), perform outlier detection, and execute feature engineering. It's an all-in-one tool to assist in the initial phases of the data science pipeline.

Features
1. Data Summary
Data Overview: Provides a summary of the dataset, including the number of rows and columns, data types, and basic statistics.
Missing Values: Identifies columns with missing values and their respective counts.
2. Exploratory Data Analysis (EDA)
Plot Selection: Users can choose from a variety of plots to visualize their data:
Scatter Plot: For visualizing the relationship between two numeric variables and spotting outliers.
Histogram: To view the distribution of a single numeric variable.
Box Plot: Summarizes data distribution and highlights outliers.
Violin Plot: Combines box plot and density plot to show data distribution and density.
3. Outlier Detection and Visualization
Users can select specific numeric columns to detect and visualize outliers using the available plot types.
4. Feature Engineering
Handling Missing Values: Impute missing values in numeric columns using strategies like mean, median, or mode.
Encoding Categorical Variables: Convert categorical variables into numeric format using techniques such as one-hot encoding.
5. Data Preprocessing
Scaling: Standardize numeric features using StandardScaler or MinMaxScaler.
Feature Selection: Select relevant features for model building based on correlation analysis.
