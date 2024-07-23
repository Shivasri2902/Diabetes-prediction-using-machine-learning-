from flask import Flask, render_template, request
import model_trained  # Ensure model_trained.py is in the same directory or adjust the path

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        smoking_history = int(request.form['smoking_history'])
        bmi = float(request.form['bmi'])
        hba1c_level = float(request.form['hba1c_level'])
        blood_glucose_level = float(request.form['blood_glucose_level'])

        # Prepare feature vector for the model
        features = [[gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level]]
        
        # Log the features for debugging
        print(f"Features: {features}")

        prediction = model_trained.predict(features)

        # Log the prediction for debugging
        print(f"Prediction: {prediction}")

        if prediction == 1:
            result_message = "OOPS!! YOU HAVE DIABETES "
        else:
            result_message = "DON'T WORRY YOU DON'T HAVE DIABETES."

    except Exception as e:
        result_message = f"An error occurred: {e}"

    return render_template('result.html', result_message=result_message)

if __name__ == '__main__':
    app.run(debug=True)
