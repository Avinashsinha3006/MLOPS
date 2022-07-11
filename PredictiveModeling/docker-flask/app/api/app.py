from flask import Flask, jsonify, request
from preprocess import preprocess
import json
#from flask import request
#from utilities import predict_pipeline
import pickle

app = Flask(__name__)

#@app.route('/')
#def index():
#    return "http://0.0.0.0:5000/predict"




@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        print("predict--------------------------------------------")

        try:
            data = request.json
            test_x = preprocess(data)
            #print(test_x)
            with open('models/finalized_model.sav', 'rb') as f:
                rf_model  = pickle.load(f)
            #rf_model = pickle.load(open('models/finalized_model.sav', 'rb'))
            print("1")
            predictions = rf_model.predict(test_x)


            result  = json.dumps({'result': predictions.tolist()})



        except Exception as e:
            result = jsonify({'error': str(e)})
        return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)