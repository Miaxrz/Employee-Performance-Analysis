from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open(r'random-forest_model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_performance():
    EmpLastSalaryHikePercent = float(
        request.form.get('EmpLastSalaryHikePercent'))
    EmpEnvironmentSatisfaction = int(
        request.form.get('EmpEnvironmentSatisfaction'))
    YearsSinceLastPromotion = int(request.form.get('YearsSinceLastPromotion'))

    # prediction
    result = model.predict(np.array(
        [EmpLastSalaryHikePercent, EmpEnvironmentSatisfaction, YearsSinceLastPromotion]).reshape(1, 3))
    if result[0] == 1:
        result = 'Low'
    elif result[0] == 2:
        result = 'Good'
    elif result[0] == 3:
        result = 'Excellent'
    elif result[0] == 4:
        result = 'Outstanding'
    else:
        result = 'None'

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
