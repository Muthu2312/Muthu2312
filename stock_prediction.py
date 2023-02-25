#Import required Libraries
import streamlit as st
from streamlit_option_menu import option_menu
import datetime as dt
import yfinance as yf
import time
import requests
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import numpy


selected=option_menu(
        menu_title='Main Menu',
        options= ['Search','Basic Info','Graphical Analysis','Prediction and Analysis'],
        icons=['search','info-circle-fill','graph-up','activity'],
        default_index=0,
        orientation='horizontal',
        styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "30px"}, 
        "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"}
        }
    )
START=  dt.date(2021, 1, 1)
END =  dt.datetime.today()

# def load_data(ticker):
#     data = yf.download(ticker, START, END)
#     data.reset_index(inplace=True)
#     return data
@st.cache_data
@st.cache_resource
def search_symbol(stock_name):
    # create sample dataframe
    df = pd.read_csv('stock.csv')
    df.dropna(inplace=True)
    return df


# load the stock symbols and names

START=  dt.date(2021, 1, 1)
END =  dt.datetime.today()
df = pd.read_csv('stock.csv')
df.dropna(inplace=True)
@st.cache_data
@st.cache_resource
def search_symbol(stock_symbol):
    df = pd.read_csv('stock.csv')
    df.dropna(inplace=True)
    # find the stock name for the given symbol
    #stock_name = df[df['Symbol'] == stock_symbol]['Name'].iloc[0]
    # download data for the stock symbol
    data = yf.download(stock_symbol,START,END)
    data.reset_index(inplace=True)
    # st.header(f"{stock_name} ({stock_symbol}) Stock Price")
    return data

search_term1 = st.text_input("Enter a stock symbol:")
search_term = search_term1.upper()
stock_name = df[df['Symbol'] == search_term]['Name'].iloc[0]
data1=search_symbol(search_term)



if selected=='Basic Info':
    if search_term:
        if search_term in df['Symbol'].values:
            data1=search_symbol(search_term)
            st.header(f"{stock_name} ({search_term}) Stock Price")
            st.experimental_data_editor(data1)
        else:
            st.write(f"No stock found for symbol '{search_term}'")
    
if selected=='Graphical Analysis':
    options = ['Option 1', 'Option 2', 'Option 3','Option 4','Option 5','Option 6']
    selected_options = st.selectbox('Select options:', options)
    if selected_options=='Option 1':
                fig = make_subplots(rows=1, cols=1)
                fig.add_trace(go.Candlestick(x=data1['Date'],
                        open=data1['Open'], high=data1['High'],
                        low=data1['Low'], close=data1['Close']),
                        row=1, col=1)
                fig.update_layout(
            title='Animated Candlestick Chart',
            yaxis_title='Price',
            xaxis_rangeslider_visible=True)
                st.plotly_chart(fig, use_container_width=True,renderer='webgl')
                agree = st.checkbox('Explanation')
                if agree:
                    st.write('A candlestick chart is a financial chart that typically shows price movements of currency, securities, or derivatives. It looks like a candlestick with a vertical rectangle and a wick at the top and bottom.\nThe top and bottom of the candlestick show open and closed prices.\n The top of the wick shows the high price, and the bottom of the wick shows the low price.')
                    st.write('**Candlestick charts show a range of information:**')
                    st.write('-Open Price.')
                    st.write('-Close Price.')
                    st.write('-Highest Price.')
                    st.write('-Lowest Buy Price.')
                    st.write('-Patterns and Trends in Share Prices.')
                    st.write('-Emotions of Trades.')


    if selected_options=='Option 2':
        trace_close = go.Scatter(
            x=data1['Date'],
            y=data1['Close'],
            name='Closing Price')
        layout = go.Layout(
            title=f'Trend of Closing Prices',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'})
        fig2 = go.Figure(data=[trace_close], layout=layout) 
        fig2.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([
        dict(count=1, label="1m", step="month", stepmode="backward"),
        dict(count=6, label="6m", step="month", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(step="all")])))
        st.plotly_chart(fig2,renderer='webgl')
        agree1 = st.checkbox('Explanation')
        if agree1:
            st.write('-A closing price trendline is a graphical representation of a closing price trend over time.')
            st.write('-The closing price is the final price at which a security or asset is traded at the end of the trading day.')
            st.write('-A trendline is a straight line drawn on a chart to indicate the direction of a trend or the general direction of closing price movement.')
            st.write('-If the trendline slopes upwards, it indicates an uptrend, which means that the closing prices have been increasing over time.')
            st.write('-Conversely, if the trendline slopes downwards, it indicates a downtrend, which means that the closing prices have been decreasing over time.')
            st.write('-Traders and investors use closing price trendlines to identify the overall trend of an asset and make informed decisions about whether to buy, sell or hold that asset.')
            st.write('-Its Important to note that trendlines are based on historical data and may not always accurately predict future price movements.')
    
    if selected_options=='Option 3':
        fig3 = px.box(data1, x=data1['Date'].dt.year, y='Close', points='all', title='Closing Prices by Year')
        st.plotly_chart(fig3,renderer='webgl')
         
