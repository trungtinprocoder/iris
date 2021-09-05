import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import pickle

iris =pd.read_csv('./iris.csv',delimiter=',')
X = iris.iloc[:,0:4]
y = iris.iloc[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = tree.DecisionTreeClassifier(criterion="gini")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
pickle.dump(model,open("model.pkl","wb"))

from flask import Flask
from flask_cors import CORS, cross_origin
from flask import render_template
from flask import request
import numpy as np
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
@cross_origin(origin='*')
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text ="{0}".format(prediction))

#start backend
if __name__== '__main__':
    app.run(debug=True)
