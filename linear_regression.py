import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

'''
This code uses a Linear Regression model, but will also show classification metrics (recall, f1-score, support).
To do that, we need a classification target. We will define a binary "success" threshold based on the number of listeners.
If predicted listeners > 10,000, we consider it "Successful" (1), otherwise "Not Successful" (0).

Note: The regression model outputs log_listeners_lastfm. We will convert predictions back to the original scale 
using expm1() and then apply the threshold.
'''

def load_data(file_path):
    data = pd.read_csv(file_path)
    
    # Define log-transformed listeners
    data['log_listeners_lastfm'] = np.log1p(data['listeners_lastfm'])
    data['filtered_tags'] = data['filtered_tags'].apply(eval)
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(data['filtered_tags'])

    le = LabelEncoder()
    countries_encoded = le.fit_transform(data['country_lastfm'])

    X = pd.concat([
        pd.DataFrame(genres_encoded, columns=mlb.classes_),
        pd.DataFrame(countries_encoded, columns=['country'])
    ], axis=1)

    y = data['log_listeners_lastfm'].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y, mlb, le, scaler, data

def split_data(X, y, test_size=0.3, val_size=0.5, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
    return X_train, X_test, X_val, y_train, y_test, y_val

if __name__ == "__main__":
    file_path = "filtered_data.csv"
    success_threshold = 5_000  # Threshold for success classification
    X, y, mlb, le, scaler, data = load_data(file_path)
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X, y)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    train_pred = lr.predict(X_train)
    test_pred = lr.predict(X_test)

    # Convert the log predictions back to original scale
    test_pred_listeners = np.expm1(test_pred)
    y_test_listeners = np.expm1(y_test)

    # Binary classification based on threshold
    test_pred_class = (test_pred_listeners > success_threshold).astype(int)
    y_test_class = (y_test_listeners > success_threshold).astype(int)

    ''' # Print classification metrics
    print("\nClassification Report:")
    report = classification_report(y_test_class, test_pred_class, target_names=["Not Successful", "Successful"])
    print(report)

    # Plot predictions vs actual for regression
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, test_pred, s=10, alpha=0.5)
    plt.xlim(8, 16)  # Focus on the range where most data resides
    plt.ylim(8, 16)
    plt.xlabel('Actual Log Listeners')
    plt.ylabel('Predicted Log Listeners')
    plt.title('Linear Regression Predictions vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.show()'''
