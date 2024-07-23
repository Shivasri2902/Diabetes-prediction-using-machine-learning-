import pickle

# Load the trained model
with open('diabetes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict(features):
    return model.predict(features)[0]  # Ensure this returns the first prediction
