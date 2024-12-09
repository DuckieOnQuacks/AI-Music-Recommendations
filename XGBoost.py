import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message=".*Falling back to prediction using DMatrix.*")
 

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

def predict_success(model, X_train, le, mlb, scaler, data, country, genres):
    # Encode country
    try:
        country_encoded = le.transform([country])[0]
    except ValueError:
        print("Invalid country. Please try again.")
        return

    # Encode genres
    genres_set = set(genres)
    valid_genres = set(mlb.classes_)
    if not genres_set.issubset(valid_genres):
        print(f"Invalid genres. Valid genres are: {', '.join(valid_genres)}")
        return

    genres_encoded = mlb.transform([genres])[0]

    # Combine inputs
    user_input = np.zeros(X_train.shape[1])
    user_input[:len(genres_encoded)] = genres_encoded  # Set genre encoding
    user_input[len(genres_encoded)] = country_encoded  # Set country encoding

    user_input[-1] = data['listeners_lastfm'].mean()

    # Normalize the input using the same scaler
    user_input_scaled = scaler.transform([user_input])

    duser_input = xgb.DMatrix(user_input_scaled, device='cuda')

    success_prob = model.predict_proba(duser_input)[0][1]
    success_rate = success_prob * 100
    print(f"Predicted Success Rate: {success_rate:.2f}%")



if __name__ == "__main__":

    file_path = "filtered_data.csv"
    X, y, mlb, le, scaler, feature_names, data = load_data(file_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    bst = xgb.XGBClassifier(n_estimators=501, max_depth=8, learning_rate=0.04, gamma=1, subsample=0.6, objective='binary:logistic', tree_method='hist', device='cuda')

    scores = cross_val_score(bst, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy scores: {scores}")
    print(f"Mean cross-validation accuracy: {scores.mean()}")
    
    bst.fit(X_train, y_train)
    
    print("Train Evaluation:")
    y_train_pred = bst.predict(X_train)
    print(classification_report(y_train, y_train_pred, target_names=['Not Successful', 'Successful']))

    print("Test Evaluation:")
    y_test_pred = bst.predict(X_test)
    print(classification_report(y_test, y_test_pred, target_names=['Not Successful', 'Successful']))
    
    importance = bst.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    top_n = 20
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'][:top_n], feature_importance_df['Importance'][:top_n])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title('Top 20 Features by Importance')
    plt.show()
    
    y_prob = bst.predict_proba(X_test)[:, 1]
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
    plt.show()

    while True:
        country = input("\nEnter the country (or type 'exit' to quit): ").strip()
        if country.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break

        genres = input("Enter genres (comma-separated): ").strip().split(',')

        # Predict success rate
        predict_success(bst, X_train, le, mlb, scaler, data, country, genres)
