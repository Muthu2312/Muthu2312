import streamlit as st
from datetime import date
import plotly.express as px
import yfinance as yf
from plotly import graph_objs as go
import numpy as np
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu

import datetime as dt
selected=option_menu(
        menu_title='Stock Market Prediction',
        options= ['About','Application'],
        icons=['buildings-fill','clipboard-data-fill'],
        default_index=0,
        orientation='horizontal',
        styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "30px"}, 
                "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "green"}
        }
        )
data=pd.read_csv('stock.csv')
data.dropna(inplace=True)
START=  dt.date(2021, 1, 1)
END =  dt.datetime.today()

#tab2, tab3,tab4 = st.tabs(["Data-set", "Data Visualization", "Predictions"])
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, END)
    data.reset_index(inplace=True)
    return data


# selected_stock = st.text_input('Enter your stock')
# df = load_data(selected_stock)
# if selected_stock in data['Symbol'].str.upper().values:
#     stock_name = data.loc[data['Symbol'].str.upper() == selected_stock, 'Name'].values[0]
#     st.write(f'The name of the stock with symbol {selected_stock} is {stock_name}.')

if selected=='Application':
    data=pd.read_csv('stock.csv')
    data.dropna(inplace=True)
    selected_stock = st.text_input('Enter your stock')
    selected_stock1=selected_stock.upper()
    df = load_data(selected_stock1)
    stock_name = data.loc[data['Symbol'].str.upper() == selected_stock, 'Name'].values[0]
    if selected_stock1 in data['Symbol'].str.upper().values:
        stock_name = data.loc[data['Symbol'].str.upper() == selected_stock1, 'Name'].values[0]
        st.header(f'The name of the stock with symbol {selected_stock1} is {stock_name}.')
    else:
        st.write(f'Stock symbol {selected_stock} not found.')
    tab2, tab3,tab4 = st.tabs(["Data-set", "Data Visualization", "Predictions"])
    with tab2:
        data_load_state = st.text('Loading data...')
        st.write("You have entered", selected_stock)
        st.experimental_data_editor(df)  
        data_load_state.text('Loading data... done!')


    df1=df.copy()
    df1['Difference']=df1['Close']-df1['Open']

    with tab3:
        fig = go.Figure(data=[go.Candlestick(x=df["Date"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"])])
        st.plotly_chart(fig)

        fig2=px.line(df, x="Date", y="Close", title='Trend of Closing price')
        fig2.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")])))

        fig3=px.line(df, x="Date", y="Adj Close", title='Trend of Adj Close')
        fig3.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")])))

        fig4=px.line(df, x="Date", y="Volume", title='Trend in Volume')
        fig4.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")])))

        fig5=px.bar(df1, x=df1["Date"], y="Difference", title='Rise and Drop for Each Day')
        fig5.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")])))
        st.plotly_chart(fig2)
        st.plotly_chart(fig3)
        st.plotly_chart(fig4)
        st.plotly_chart(fig5)


    df1=df.reset_index()['Close']
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

    training_size=int(len(df1)*0.7)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

    import numpy
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


    with tab4:
        model=Sequential()
        model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
        model.add(LSTM(50,return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')

        model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=20,batch_size=64,verbose=1)
        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)

        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)



        x_input=test_data[len(test_data)-100:].reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()

        lst_output=[]
        n_steps=100
        i=0
        while(i<30):
            if(len(temp_input)>100):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                #print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = model.predict(x_input, verbose=0)
                st.write("For Day {}, the predicted output is {}".format(i,scaler.inverse_transform(yhat)))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                # print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                #print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1
