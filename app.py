from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        Item_MRP = float(request.form['Item_MRP'])
        Item_Weight = float(request.form['Item_Weight'])
        Item_Visibility = float(request.form['Item_Visibility'])
        Item_Fat_Content = int(request.form['Item_Fat_Content'])
        Item_Type = int(request.form['Item_Type'])
        Outlet_Size = int(request.form['Outlet_Size'])
        Outlet_Location_Type = int(request.form['Outlet_Location_Type'])
        Outlet_Type = int(request.form['Outlet_Type'])

        # Arrange the features as per model training
        features = np.array([[Item_Weight, Item_Visibility, Item_Type, Item_Fat_Content,
                              Item_MRP, Outlet_Size, Outlet_Location_Type, Outlet_Type]])

        # Predict
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=round(prediction, 2))
    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
