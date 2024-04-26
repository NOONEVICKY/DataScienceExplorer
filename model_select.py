import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

class ModelSelection:
    def __init__(self, data: pd.DataFrame, target_var: str):
        self.data = data
        self.target_var = target_var
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            data.drop(columns=[target_var]), data[target_var], test_size=0.2, random_state=42
        )
        self.mods = {}
        self.target_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.feature_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Support Vector Machine': SVC(),
            'Decision Tree': DecisionTreeClassifier()
        }

    def preprocess_data(self):
        self.y_train_processed = self.target_pipeline.fit_transform(self.y_train.values.reshape(-1, 1)).ravel()
        self.X_train_scaled = self.feature_pipeline.fit_transform(self.x_train)
        self.X_test_scaled = self.feature_pipeline.transform(self.x_test)

    def train_and_test_models(self):
        for model_name, model in self.models.items():
            model.fit(self.X_train_scaled, self.y_train_processed)
            y_pred = model.predict(self.X_test_scaled)
            cm = confusion_matrix(self.y_test, y_pred)

            self.mods[model_name] = {
                'model': model,
                'accuracy': model.score(self.X_test_scaled, self.y_test),
                'confusion_matrix': cm,
                'classification_report': classification_report(self.y_test, y_pred)
            }

        return self.mods
