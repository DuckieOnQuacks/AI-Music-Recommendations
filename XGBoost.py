import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc


# Load and preprocess data
def load_data(file_path, success_threshold=10_000):
    data = pd.read_csv(file_path)

    # Define success as a binary classification
    data['success'] = (data['listeners_lastfm'] > success_threshold).astype(int)
    
    data['filtered_tags'] = data['filtered_tags'].apply(eval)
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(data['filtered_tags'])

    # Process countries
    le = LabelEncoder()
    countries_encoded = le.fit_transform(data['country_lastfm'])

    X = pd.concat([  
        pd.DataFrame(genres_encoded, columns=mlb.classes_),
        pd.DataFrame(countries_encoded, columns=['country'])
    ], axis=1)

    feature_names = X.columns.tolist()

    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    y = data['success'].values

    return X, y, mlb, le, scaler, feature_names, data

def predict_success(model, mlb, le, scaler, country, genres):
    # Encode country
    try:
        country_encoded = le.transform([country])[0]
    except ValueError:
        return "Invalid country. Please try again."

    # Encode genres
    genres_set = set(genres)
    valid_genres = set(mlb.classes_)
    if not genres_set.issubset(valid_genres):
        return f"Invalid genres. Valid genres are: {', '.join(valid_genres)}"

    genres_encoded = mlb.transform([genres])[0]

    # Combine inputs
    user_input = np.zeros(len(mlb.classes_) + 1)  # +1 for the country feature
    user_input[:len(genres_encoded)] = genres_encoded  # Set genre encoding
    user_input[-1] = country_encoded  # Set country encoding

    # Normalize the input using the same scaler
    user_input_scaled = scaler.transform([user_input])

    # Use model to predict probabilities directly
    success_prob = model.predict_proba(user_input_scaled)[0][1]
    success_rate = success_prob * 100
    return f"{success_rate:.2f}%"

def run_xgboost(genres, country):
    file_path = "filtered_data.csv"
    X, y, mlb, le, scaler, feature_names, data = load_data(file_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize XGBoost Classifier
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    ''' # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {scores}")
    print(f"Mean cross-validation accuracy: {scores.mean():.2f}")

    # Evaluate on training and test data
    print("Train Evaluation:")
    y_train_pred = model.predict(X_train)
    print(classification_report(y_train, y_train_pred, target_names=['Not Successful', 'Successful']))

    print("Test Evaluation:")
    y_test_pred = model.predict(X_test)
    print(classification_report(y_test, y_test_pred, target_names=['Not Successful', 'Successful']))

    # Feature importance plot
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'][:20], feature_importance_df['Importance'][:20])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title('Top 20 Features by Importance')
    plt.show()

    # ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()'''

    # Predict success rate
    return predict_success(model, mlb, le, scaler, country, genres)

