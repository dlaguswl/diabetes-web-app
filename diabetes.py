import os
import numpy as np
import pandas as pd
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from bootstrap_flask import Bootstrap5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

class LabForm(FlaskForm):
    preg = StringField('Pregnancies', validators=[DataRequired()])
    glucose = StringField('Glucose', validators=[DataRequired()])
    blood = StringField('Blood pressure', validators=[DataRequired()])
    skin = StringField('Skin thickness', validators=[DataRequired()])
    insulin = StringField('Insulin', validators=[DataRequired()])
    bmi = StringField('BMI', validators=[DataRequired()])
    dpf = StringField('DPF score', validators=[DataRequired()])
    age = StringField('Age', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        X_test = np.array([[float(form.preg.data),
                            float(form.glucose.data),
                            float(form.blood.data),
                            float(form.skin.data),
                            float(form.insulin.data),
                            float(form.bmi.data),
                            float(form.dpf.data),
                            float(form.age.data)]])

        print(X_test.shape)
        print(X_test)

        csv_path = os.path.join(BASE_DIR, 'diabetes.csv')
        data = pd.read_csv(csv_path, sep=',')

        X = data.values[:, 0:8]
        y = data.values[:, 8]

        scaler = MinMaxScaler()
        scaler.fit(X)

        X_test = scaler.transform(X_test)

        model_path = os.path.join(BASE_DIR, 'pima_model.keras')
        model = keras.models.load_model(model_path)

        prediction = model.predict(X_test)
        res = prediction[0][0]
        res = np.round(res, 2)
        res = float(np.round(res * 100))

        return render_template('result.html', res=res)

    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
