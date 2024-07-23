import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import pickle

# Load your dataset
data = pd.read_csv("C:\\Users\\shiva\\Downloads\\Diabetes-Prediction-using-machine-learning-main\\Diabetes-Prediction-using-machine-learning-main\\diabetes_prediction_dataset.csv")  # Replace with your actual data file

# Define features and target
X = data[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = data['diabetes']  # Replace with your actual target column name

# Encode categorical variables if necessary
X['gender'] = X['gender'].apply(lambda x: 1 if x.lower() == 'male' else 0)
X['smoking_history'] = X['smoking_history'].apply(lambda x: 1 if x.lower() == 'yes' else 0)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)
DT = DecisionTreeClassifier()
DT.fit(X_train_encoded, y_train)

# Save the trained model
with open('diabetes_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Test the model (optional)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Test accuracy:", accuracy_score(y_test, DT.predict(X_test_encoded)))
print("Training accuracy:", accuracy_score(y_train, DT.predict(X_train_encoded)))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

