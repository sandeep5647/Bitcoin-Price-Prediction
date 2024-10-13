import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model   # type: ignore
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta  

# Load Model
# model = load_model(r'E:\MCA\Bitcoin_Price_Prediction\Bitcoin_Price_Prediction.keras')  #for local machine
model = load_model('Bitcoin_Price_Prediction.keras')   # For deployment

st.header('Bitcoin Price Prediction Model')

st.markdown(f"""
**Disclaimers:**  
- The latest available data is from the last trading day, as financial markets are closed on weekends or holidays.
- This application is for learning and educational purposes only. Please **do not make any investment decisions** based on these predictions.
""")

st.subheader('Bitcoin Price Trends from the Last Year Until Now (USD)')
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

# Fetching Bitcoin data
data = pd.DataFrame(yf.download('BTC-USD', start=start_date, end=end_date))
data = data.drop(columns=['Adj Close'])
data = data.reset_index()

st.write(data.iloc[::-1])  # Display in reverse order

# Bitcoin Line Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Bitcoin Price'))
fig.update_layout(title='Bitcoin Price Over Time (USD)', xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(fig)

# Preprocessing for prediction
raw_data = data[['Date', 'Close']]
test_data = raw_data[-200:]  # lastest 200 

scaler = MinMaxScaler(feature_range=(0, 1))
test_data_scaled = scaler.fit_transform(test_data[['Close']])

# Preparing the data for prediction
base_days = 100
x = []
y = []
for i in range(base_days, test_data_scaled.shape[0]):
    x.append(test_data_scaled[i - base_days:i])
    y.append(test_data_scaled[i, 0])

x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# Prediction using model
st.subheader('Predicted vs Original Prices')
pred = model.predict(x)
pred = scaler.inverse_transform(pred)
preds = pred.reshape(-1, 1)
ys = scaler.inverse_transform(y.reshape(-1, 1))

# Conversion into DataFrame
preds = pd.DataFrame(preds, columns=['Predicted Price'])
ys = pd.DataFrame(ys, columns=['Original Price'])
result = pd.concat([test_data['Date'].iloc[base_days:].reset_index(drop=True), preds, ys], axis=1)

st.write(result.iloc[::-1])

# Predicted vs Original Prices Chart
pred_chart = go.Figure()
pred_chart.add_trace(go.Scatter(x=result['Date'], y=result['Predicted Price'], mode='lines', name='Predicted Price'))
pred_chart.add_trace(go.Scatter(x=result['Date'], y=result['Original Price'], mode='lines', name='Original Price'))
pred_chart.update_layout(title='Predicted vs Original Bitcoin Prices (USD)', xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(pred_chart)

# Future price prediction
m = y
z = []
future_days = 10
for i in range(base_days, len(m) + future_days):
    m = m.reshape(-1, 1)
    inter = [m[-base_days:, 0]]
    inter = np.array(inter)
    inter = np.reshape(inter, (inter.shape[0], inter.shape[1], 1))
    pred = model.predict(inter)
    m = np.append(m, pred)
    z = np.append(z, pred)

st.subheader('Predicted Future Bitcoin Price')
z = np.array(z)
z = scaler.inverse_transform(z.reshape(-1, 1))

last_date = pd.to_datetime(test_data['Date'].iloc[-1])
future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
future_dates = [date.strftime('%d-%b') for date in future_dates]

# Predicted future price chart
future_chart = go.Figure()
future_chart.add_trace(go.Scatter(x=future_dates, y=z[-future_days:, 0], mode='lines', name='Predicted Future Price'))
future_chart.update_layout(title='Predicted Future Bitcoin Prices (USD)', xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(future_chart)
