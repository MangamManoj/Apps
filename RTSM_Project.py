import pandas as pd
import numpy as np
import pandas_datareader as data
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2019-01-01'

st.title('Stock Prediction')

usr_ip = st.text_input('Enter','AAPL')
df= data.DataReader(usr_ip,"yahoo",start,end)

st.subheader('Data from 2010-2019')
st.write(df.describe())

df.reset_index(inplace=True)

df.drop(['Adj Close'],axis=1,inplace=True)

#Vizualizations
st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100 MA & 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(df.Close,'b')
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

df.shape

df_train = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
df_test = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

print(df_train.shape)
print(df_test.shape)

df_train.head()

df_test.head()

scaler = MinMaxScaler(feature_range=(0,1))

df_train_array = scaler.fit_transform(df_train)
df_train_array



#Train test splitting
x_train = []
y_train = []

for i in range(100,df_train_array.shape[0]):
    x_train.append(df_train_array[i-100: i])
    y_train.append(df_train_array[i,0])

x_train,y_train = np.array(x_train), np.array(y_train)


#Load my model
model = load_model('keras_model.h5')

past_100_days = df_train.tail(100)
final_df = past_100_days.append(df_test, ignore_index=True)

input_df = scaler.fit_transform(final_df)

x_test =[]
y_test =[]

for i in range(100, input_df.shape[0]):
    x_test.append(input_df[i-100: i])
    y_test.append(input_df[i,0])

x_test,y_test = np.array(x_test),  np.array(y_test)

y_pred = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_pred = y_pred*scale_factor
y_test = y_test*scale_factor


#Final Graph

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Actual Price')
plt.plot(y_pred,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)






