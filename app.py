import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import RobustScaler
from flask import Flask, request, render_template
import tensorflow as tf
import tensorflow.keras as keras

df = pd.read_csv("C:/Users/Thanh Tung/PycharmProjects/Diplom_master/loans.csv")
df = pd.get_dummies(df, columns=["purpose"], drop_first=True)
X = df.loc[:, df.columns != "not.fully.paid"].values
y = df.loc[:, df.columns == "not.fully.paid"].values.flatten()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=123, stratify=y)

std = RobustScaler()
std.fit(X_train)
model = keras.models.load_model('model_relu.h5')

# Tạo ứng dụng
app = Flask(__name__)


# Liên kết hàm home với URL
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/index.html')
def logo():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [float(i) for i in request.form.values()]
    features[1] = int(features[1])
    a_ = ["credit_card", "debt_consolidation", "educational", "home_improvement", "major_purchase", "small_business",
          "all_other"]
    for j in range(7):
        if features[1] == j:
            features[1] = a_[j]
    a = ["credit_card", "debt_consolidation", "educational", "home_improvement", "major_purchase", "small_business"]
    b = [0, 0, 0, 0, 0, 0]
    for i in range(len(a)):
        if a[i] == features[1]:
            b[i] = 1
    X = np.concatenate((features, b), axis=0)
    X = np.delete(X, 1)
    X = X.reshape(1, 18)
    X = std.transform(X)
    # Predict
    prediction = model.predict(X)
    output = prediction
    # Kiểm tra các giá trị đầu ra và truy xuất kết quả bằng thẻ html dựa trên giá trị
    if output == 1:
        return render_template('result.html', credit=int(features[0]),
                               purpose=features[1],
                               rate=features[2],
                               installment=int(features[3]),
                               log=features[4],
                               dti=features[5],
                               fico=int(features[6]),
                               days=int(features[7]),
                               revol=int(features[8]),
                               revol_=features[9],
                               inq=int(features[10]),
                               delinq=int(features[11]),
                               pub=int(features[12]),
                               result='Will not pay in full!')
    else:
        return render_template('result.html', credit=int(features[0]),
                               purpose=features[1],
                               rate=features[2],
                               installment=int(features[3]),
                               log=features[4],
                               dti=features[5],
                               fico=int(features[6]),
                               days=int(features[7]),
                               revol=int(features[8]),
                               revol_=features[9],
                               inq=int(features[10]),
                               delinq=int(features[11]),
                               pub=int(features[12]),
                               result='Will pay in full!')


if __name__ == '__main__':
    # Chạy ứng dụng
    app.run()
