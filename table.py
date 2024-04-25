from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score,
                             confusion_matrix, mean_absolute_error, mean_squared_error)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier





class Table:
    """
    Present data as a table class.

    Keyword arguments:
    dataset -- dataset that fetched from fetch_ucirepo
    Return: Table
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.features = dataset.features
        # Initialize the Logistic Regression model
        self.model = LogisticRegression(solver="liblinear",random_state=42 , max_iter=1000)

    def display_columns(self):
        """ Display the Column Names of the dataset """
        print("Column Names:", self.features.columns.tolist())

    def table_size(self):
        """ Display the number of rows and columns """
        print("Number of Rows:", self.features.shape[0])
        print("Number of Columns:", self.features.shape[1])

    def display_rows(self, number=None, reverse=False):
        """
        Display the rows of the dataset.
        
        Keyword arguments:
        number -- number of rows to display (default is None, which displays all rows)
        reverse -- boolean flag to display the last rows instead of the first (default is False)
        """
        if number is None:
            number = self.features.shape[0]
        order = "Last" if reverse else "First"
        method = "tail" if reverse else "head"
        print(f"{order} {number} Rows:")
        print(getattr(self.features, method)(number))

    def statistics(self):
        """ Display basic statistics of the dataset """
        print("Basic Statistics:")
        print(self.features.describe())

    def missing_values(self):
        """ Find any missing values in the dataset """
        return self.features.isnull().sum()

        
    def display_missing_values(self):
        """ Find and display any missing values in the dataset """
        print("Missing Values:")
        print(self.missing_values())


    def handle_missing_values(self):
        """ handle missing values in the dataset """
        # using most_frequent to the data because the values should be between 0 and 4
        # this will ensure that the values not be decimal and within the range 
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(self.features)
        #make a new dataFrame with no missing values
        idf=pd.DataFrame(imputer.transform(self.features) , columns=self.features.columns , index=self.features.index)
        # replace old data with the valid one
        self.features = idf
        


    def splitData(self):
        # Split the data into training and testing sets
        # The test_size parameter determines the proportion of the dataset to include in the test split
        # random_state is set to a fixed number to ensure reproducibility
        self.X_train,\
        self.X_test,  \
        self.y_train,  \
        self.y_test = train_test_split(self.features, self.dataset.targets, train_size=0.8, random_state=42)
        
        print("Data splitted into 80% taring and 20% testing")
        
    def build_model(self):
        from sklearn.pipeline import Pipeline
        
        """ Build and train the Random Forest Classifier model with data scaling"""
        # Create a pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('model', self.model)
        ])

        # Train the model
        pipeline.fit(self.X_train, self.y_train.values.ravel())
        
        # Predict on the training data (to check for overfitting)
        self.train_predictions = pipeline.predict(self.X_train)
        print("Training Accuracy:", accuracy_score(self.y_train, self.train_predictions))

        # Predict on the testing data
        self.test_predictions = pipeline.predict(self.X_test)
        print("Test Accuracy:", accuracy_score(self.y_test, self.test_predictions))

    def fine_tune(self):
        # Define the parameter grid
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2']
        }

        # Initialize the Grid Search with cross-validation
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy')

        # Fit the Grid Search to the data
        grid_search.fit(self.X_train , self.y_train.values.ravel())

        # Get the best model
        best_model = grid_search.best_estimator_

        # Evaluate the best model
        best_predictions = best_model.predict(self.X_test)
        print("Best Model Test Accuracy:", accuracy_score(self.y_test, best_predictions))
        print("Best Model Parameters:", grid_search.best_params_)
        
        
    def evaluate(self):

        print("Training Accuracy:", accuracy_score(self.y_train, self.train_predictions))
        print("Test Accuracy:", accuracy_score(self.y_test, self.test_predictions))
        
        # Recall: Proportion of actual positives correctly predicted
        recall = recall_score(self.y_test, self.test_predictions, average='macro')
        print(f"Recall: {recall:.2f}")

        # Precision: Proportion of positive identifications that were actually correct
        precision = precision_score(self.y_test, self.test_predictions, average='macro' , zero_division=0)
        print(f"Precision: {precision:.2f}")

        # F1 Score: Harmonic mean of precision and recall
        f1 = f1_score(self.y_test, self.test_predictions, average='macro')
        print(f"F1 Score: {f1:.2f}")

        # Confusion Matrix: Shows the model's performance with actual vs predicted values
        conf_matrix = confusion_matrix(self.y_test,self. test_predictions)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Mean Absolute Error (MAE): Average absolute difference between actual and predicted
        mae = mean_absolute_error(self.y_test, self.test_predictions)
        print(f"Mean Absolute Error: {mae:.2f}")

        # Root Mean Squared Error (RMSE): Square root of the average squared differences
        rmse = np.sqrt(mean_squared_error(self.y_test, self.test_predictions))
        print(f"Root Mean Squared Error: {rmse:.2f}")