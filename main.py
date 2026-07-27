from flask import Flask, render_template,request
import pickle

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    
    # get data logic
    SepalLengthCm = float(request.form['sepal_length'])
    SepalWidthCm = float(request.form['sepal_width'])
    PetalLengthCm = float(request.form['petal_length'])
    PetalWidthCm = float(request.form['petal_width'])
    user_data = [[SepalLengthCm,	SepalWidthCm,	PetalLengthCm,	PetalWidthCm]]
    print(user_data)

    # model initiate
    with open('artifacts/iris_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # make prediction
    prediction = model.predict(user_data)[0]
    print(prediction)

    return render_template('index.html', html_prediction=prediction)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)