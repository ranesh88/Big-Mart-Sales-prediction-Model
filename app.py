import numpy as np
import pandas as pd
import pickle
from flask import Flask,request,render_template
from sklearn.preprocessing import StandardScaler

# loading model
model = pickle.load(open('lr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def index():
    return  render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    Item_Weight  = request.form['Item_Weight']
    Item_Fat_Content = request.form['Item_Fat_Content']
    Item_Visibility = request.form['Item_Visibility']
    Item_Type = request.form['Item_Type']
    Item_MRP = request.form['Item_MRP']
    Outlet_Establishment_Year = request.form['Outlet_Establishment_Year']
    Outlet_Size = request.form['Outlet_Size']
    Outlet_Location_Type = request.form['Outlet_Location_Type']
    Outlet_Type = request.form['Outlet_Type']

    features = np.array([[Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Establishment_Year,Outlet_Size,Outlet_Location_Type,Outlet_Type]],dtype=object)
    transformed_features = preprocessor.transform(features)
    prediction = model.predict(transformed_features).reshape(1,-1)

    return render_template('index.html',output = prediction[0])


# main python
if __name__ == "__main__":
    app.run(debug=True)