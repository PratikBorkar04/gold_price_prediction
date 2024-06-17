from flask import Flask, request, render_template
import numpy as np
import pickle
import time

app = Flask(__name__)
model = pickle.load(open("random_forest_model.pkl", "rb"))

def process_features(input_features):
    # Implement your feature processing logic here if needed
    return input_features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the form
    if request.method == 'POST':
        open_val = int(request.form['Open'])
        high_val = int(request.form['High'])
        low_val = int(request.form['Low'])
        close_val = int(request.form['Close'])
        volume_val = int(request.form['Volume'])
        sp_close_val = int(request.form['SP_close'])
        dj_close_val = int(request.form['DJ_close'])
        of_price_val = int(request.form['OF_Price'])
        os_price_val = int(request.form['OS_Price'])
        usdi_price_val = int(request.form['USDI_Price'])
        gdx_close_val = int(request.form['GDX_Close'])
        of_volume_val = int(request.form['OF_Volume'])

        # Create a numpy array for the input features
        input_features = np.array([[open_val, high_val, low_val, close_val, volume_val, sp_close_val,
                                    dj_close_val, of_price_val, os_price_val, usdi_price_val, gdx_close_val,
                                    of_volume_val]])

        # Record the start time
        start_time = time.time()

        # Process the input features
        processed_features = process_features(input_features)

        # Make the prediction using the model
        prediction = model.predict(processed_features)

        # Record the end time
        end_time = time.time()

        # Calculate the inference time
        inference_time = "{:.2f}".format(end_time - start_time)

        return render_template('index.html', prediction=prediction[0], inference_time=inference_time)
    else:
        return "Method Not Allowed"

if __name__ == "__main__":
    app.run(debug=True)
