from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = pd.read_csv("MicrosoftStock.csv")
print(data.head())
print(data.info())

#plots
#plt 1
# plt.figure(figsize=(12,6))
# plt.plot(data['date'],data['open'],label="Open",color="blue")
# plt.plot(data['date'],data['close'],label="closed",color="red")
# plt.title("open-close price over time")
# plt.legend()
# #plt.show()

# #plt 2
# plt.figure(figsize=(12,6))
# plt.plot(data['date'],data['volume'],label="volume",color="orange")
# plt.title("stock volume over time")
# #plt.show()

# #plt3 to check the correlation between numeric columns
# numeric_cols = data.select_dtypes(include=["int64","float64"])
# plt.figure(figsize=(8,6))
# sns.heatmap(numeric_cols.corr(),annot=True,cmap="coolwarm")
# plt.title("Feature correlation heatmap")
# plt.show()

#convert into datetime type from object type
data['date'] = pd.to_datetime(data['date'])

prediction = data.loc[
    (data['date'] > datetime(2013,1,1)) &
    (data['date'] < datetime(2018,1,1))
]

plt.figure(figsize=(12,6))
plt.plot(data['date'],data['close'],color="red")
plt.xlabel("date")
plt.ylabel("close")
plt.title("price over time")
plt.show()

stock_close = data.filter(["close"])
dataset = stock_close.values
training_data_len = int(np.ceil(len(dataset)*0.95))

scalar = StandardScaler()
scaled_data = scalar.fit_transform(dataset)

training_data = scaled_data[:training_data_len]  
X_train, y_train = [], []

for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#build the model

model = keras.models.Sequential()

model.add(keras.layers.LSTM(64,return_sequences=True,input_shape=(X_train.shape[1],1)))
model.add(keras.layers.LSTM(64,return_sequences=False))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer="adam",loss="mae",metrics=[keras.metrics.RootMeanSquaredError()])

training = model.fit(X_train, y_train, epochs=20, batch_size=32)

#prepare the test data
test_data = scaled_data[training_data_len-60:]
X_test, y_test = [], dataset[training_data_len:]

for i in range(60,len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))

predictions = model.predict(X_test)
predictions = scalar.inverse_transform(predictions)

train = data[:training_data_len]
test = data[training_data_len:]
test = test.copy()
test['Predictions'] = predictions

plt.figure(figsize = (12,8))
plt.plot(train['date'], train['close'], label="Train(Actual)", color="blue")
plt.plot(test['date'], test['close'], label="Test(Actual)", color="red")
plt.plot(test['date'], test['Predictions'], label="Predictions", color="orange")
plt.xlabel("date")
plt.ylabel("close")
plt.title("Price over time")
plt.show()



