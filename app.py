import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model   # type: ignore
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import plotly.graph_objects as go

# Load Model
# model = load_model(r'E:\MCA\Bitcoin_Price_Prediction\Bitcoin_Price_Prediction.keras')  #for local machine
model = load_model('Bitcoin_Price_Prediction.keras')   #for deployement

st.header('Bitcoin Price Prediction Model')         
st.subheader('Historical Price Data (USD)')

# Fetch Bitcoin data
data = pd.DataFrame(yf.download('BTC-USD', '2015-01-01', '2024-10-10'))
data = data.reset_index()
st.write(data)

# Plot raw Bitcoin price data using Plotly
st.subheader('Bitcoin Line Chart')

# Using Plotly for chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Bitcoin Price'))
fig.update_layout(title='Bitcoin Price Over Time (USD)', xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(fig)  # Use Plotly for the line chart

# Preprocess the data
data.drop(columns=['Date', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
train_data = data[:-100]
test_data = data[-200:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scale = scaler.fit_transform(train_data)
test_data_scale = scaler.transform(test_data)

base_days = 100
x = []
y = []
for i in range(base_days, test_data_scale.shape[0]):
    x.append(test_data_scale[i-base_days:i])
    y.append(test_data_scale[i, 0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Predict using the model
st.subheader('Predicted vs Original Prices')
pred = model.predict(x)
pred = scaler.inverse_transform(pred) 
preds = pred.reshape(-1, 1)
ys = scaler.inverse_transform(y.reshape(-1, 1))

# Convert predictions and actual values into a DataFrame
preds = pd.DataFrame(preds, columns=['Predicted Price'])
ys = pd.DataFrame(ys, columns=['Original Price'])
chart_data = pd.concat((preds, ys), axis=1)
st.write(chart_data)

# Plot predicted vs original prices
st.subheader('Predicted vs Original Prices Chart')

fig_pred_vs_original = go.Figure()
fig_pred_vs_original.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Predicted Price'], mode='lines', name='Predicted Price'))
fig_pred_vs_original.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Original Price'], mode='lines', name='Original Price'))
fig_pred_vs_original.update_layout(title='Predicted vs Original Bitcoin Prices (USD)', xaxis_title='Time', yaxis_title='Price (USD)')
st.plotly_chart(fig_pred_vs_original)

# Future price prediction
m = y
z = []
future_days = 5
for i in range(base_days, len(m) + future_days):
    m = m.reshape(-1, 1)
    inter = [m[-base_days:, 0]]
    inter = np.array(inter)
    inter = np.reshape(inter, (inter.shape[0], inter.shape[1], 1))
    pred = model.predict(inter)
    m = np.append(m, pred)
    z = np.append(z, pred)

st.subheader('Predicted Future Days Bitcoin Price')
z = np.array(z)
z = scaler.inverse_transform(z.reshape(-1, 1))

# Plot future predicted prices
fig_future = go.Figure()
fig_future.add_trace(go.Scatter(x=np.arange(len(z)), y=z[:, 0], mode='lines', name='Predicted Future Price'))
fig_future.update_layout(title='Predicted Future Bitcoin Prices (USD)', xaxis_title='Future Days', yaxis_title='Price (USD)')
st.plotly_chart(fig_future)
