import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas_datareader as data
from datetime import datetime, timedelta, date
import yfinance as yfin
yfin.pdr_override()
from keras.models import load_model
import streamlit as st
import requests
from bs4 import BeautifulSoup

import datetime

# startDate , as per our convenience we can modify
startDate = datetime.datetime(2012, 1, 1)

# endDate , as per our convenience we can modify
endDate = datetime.datetime(2022, 12, 31)

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker','AAPL')

# Use the user_input variable here
GetInformation = yfin.Ticker(user_input)
# Pass startDate and endDate to the history() function
df = GetInformation.history(start=startDate, end=endDate)


# start='2012-01-01'
# end='2022-12-31'
# st.title('Stock Trend Prediction')
# user_input = st.text_input('Enter Stock Ticker','AAPL')
# df = data.DataReader(user_input,'yahoo',start,end)



# Describing Data
st.subheader('Data from 2012-2022')
st.write(df.describe())

# Visulations
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

#Splitting Data into Training and Testing
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
print(data_train.shape)
print(data_test.shape)


# Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_train_array=scaler.fit_transform(data_train)

# # splitting data into x_train,y_train
# x_train=[]
# y_train=[]
# for i in range(100,data_train_array.shape[0]):
#     x_train.append(data_train_array[i-100 :i])
#     y_train.append(data_train_array[i,0])
# x_train


# load model
model=load_model('keras_model.h5')


# testing part


# for predicting the first 100 values we need the previous 100 values so we will append them
past_100_days=data_train.tail(100)

final_df=past_100_days.append(data_test, ignore_index=True)
# Scaling the testing data
input_data=scaler.fit_transform(final_df)
x_test=[]
y_test=[]
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


x_test , y_test = np.array(x_test),np.array(y_test)


# making predictions
y_predicted = model.predict(x_test)
scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test=y_test*scale_factor


# final graph
st.subheader('Predications vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)
