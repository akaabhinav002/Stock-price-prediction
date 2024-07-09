%%writefile demo.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
import os

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the Keras model
model_path = '/content/drive/MyDrive/keras_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file was not found at path: {model_path}")
loaded_model = load_model(model_path)

# Function to fetch latest stock market news and perform sentiment analysis
def fetch_stock_news(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news  # Fetch latest news
        company_info = ticker.info
        company_logo = company_info.get('logo_url', None)
    except ValueError:
        st.error(f"Failed to fetch news for {symbol}. Please enter a valid stock ticker.")
        return None, None

    # Perform sentiment analysis
    sentiments = []
    if news:
        for article in news:
            title = article['title']
            sentiment = TextBlob(title).sentiment.polarity
            sentiments.append((title, sentiment))
    else:
        st.write("No news found.")

    return sentiments, company_logo

# Function to fetch company information
def fetch_company_info(symbol):
    company = yf.Ticker(symbol)
    info = {
        'Name': company.info.get('shortName', 'Not available'),
        'Sector': company.info.get('sector', 'Not available'),
        'Market Cap': company.info.get('marketCap', 'Not available'),
        'PE Ratio': company.info.get('trailingPE', 'Not available'),
        'Website': company.info.get('website', 'Not available'),
    }
    return info

# Function to fetch market insights based on stock performance
def fetch_market_insights(df):
    if len(df) < 200:
        st.error("Insufficient data to analyze.")
        return {
            'reason': 'Insufficient data',
            'suggestion': 'No recommendation due to lack of data.',
        }

    # Calculate 100-day and 200-day moving averages
    ma100 = df['Close'].rolling(window=100).mean()
    ma200 = df['Close'].rolling(window=200).mean()

    # Determine trend based on moving averages
    if ma100.iloc[-1] > ma200.iloc[-1] and ma100.iloc[-2] <= ma200.iloc[-2]:
        reason = 'Golden Cross (Short-term bullish signal)'
        suggestion = 'Buy recommendation as short-term trend is bullish.'
    elif ma100.iloc[-1] < ma200.iloc[-1] and ma100.iloc[-2] >= ma200.iloc[-2]:
        reason = 'Death Cross (Short-term bearish signal)'
        suggestion = 'Sell recommendation as short-term trend is bearish.'
    elif ma100.iloc[-1] > ma200.iloc[-1]:
        reason = 'Short-term trend is above the long-term trend'
        suggestion = 'Consider holding or buying as trend appears bullish.'
    elif ma100.iloc[-1] < ma200.iloc[-1]:
        reason = 'Short-term trend is below the long-term trend'
        suggestion = 'Consider holding or selling as trend appears bearish.'

    return {
        'reason': reason,
        'suggestion': suggestion,
    }

# Set up Streamlit app title and sidebar
st.set_page_config(page_title="Stock Trend Prediction App", page_icon=":chart_with_upwards_trend:", layout='wide')
st.sidebar.title("Stock Market")

# Add some ticker options
default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
symbol = st.sidebar.text_input("Enter a Stock Ticker", value='AAPL')

# Fetch data using yfinance
start = '2010-01-01'
end = '2024-12-31'
try:
    df = yf.download(symbol, start=start, end=end)
except ValueError:
    st.sidebar.error(f"Failed to fetch data for {symbol}. Please enter a valid stock ticker.")
    st.stop()

# Fetch company information
company_info = fetch_company_info(symbol)

# Main content of the app
st.title("Stock Trend Prediction")

# Display the stock price prediction image at the top
st.image("/content/drive/MyDrive/Screenshot 2024-07-08 183536.png", caption="Stock Price Prediction", width=1000)

# Display Company Name
st.subheader(f"Company: {company_info['Name']}")

# Display Data Summary
st.subheader('Data Summary (2010 - 2024)')
st.write(df.describe())

# Today's Data
st.subheader('Today\'s Data')
today_data = df.tail(1)
st.table(today_data)

# Visualization
st.subheader("Closing Price vs Time Chart")
fig = plt.figure(figsize=(10, 6))
plt.plot(df['Close'], label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100MA and 200MA")
ma100 = df['Close'].rolling(window=100).mean()
ma200 = df['Close'].rolling(window=200).mean()
fig = plt.figure(figsize=(10, 6))
plt.plot(ma100, 'r', label='100-day MA')
plt.plot(ma200, 'g', label='200-day MA')
plt.plot(df['Close'], 'b', label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

# Splitting Data into training and testing
train = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
test = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_arr = scaler.fit_transform(train)

# Testing part
past_100days = train.tail(100)

# Concatenate past 100 days with test data
final_df = pd.concat([past_100days, test], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

# Create input sequences for prediction
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make Predictions
y_pred = loaded_model.predict(x_test)
y_pred = y_pred * scaler.scale_[0]
y_test = y_test * scaler.scale_[0]

# Plot predictions vs original prices
st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize=(10, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Fetch and display stock market news and company logo in the main content
news_sentiments, company_logo = fetch_stock_news(symbol)
if news_sentiments is not None:
    st.subheader(f"Latest News for {symbol}")
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for title, sentiment in news_sentiments:
        sentiment_label = 'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'
        if sentiment > 0:
            positive_count += 1
        elif sentiment < 0:
            negative_count += 1
        else:
            neutral_count += 1
        st.markdown(f"**{title}** - Sentiment: {sentiment_label}")

    # Plot sentiment analysis results
    sentiment_counts = [positive_count, negative_count, neutral_count]
    sentiment_labels = ['Positive', 'Negative', 'Neutral']
    fig_sentiment = plt.figure(figsize=(3, 3))
    plt.pie(sentiment_counts, labels=sentiment_labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#ff9999','#99ff99'])
    plt.gca().set_facecolor('black')  # Setting black background
    plt.axis('equal')
    st.subheader("Sentiment Analysis")
    st.pyplot(fig_sentiment)

# Display company logo if available
if company_logo:
    st.image(company_logo, caption=f"Logo of {symbol}", use_column_width=True)

# Fetch and display company information in the sidebar
st.sidebar.subheader("Company Information")
st.sidebar.write(f"**Sector:** {company_info['Sector']}")
st.sidebar.write(f"**Market Cap:** {company_info['Market Cap']}")
st.sidebar.write(f"**PE Ratio:** {company_info['PE Ratio']}")
st.sidebar.write(f"**Website:** [{company_info['Website']}]({company_info['Website']})")

# Fetch and display market insights and suggestions
insights = fetch_market_insights(df)
st.subheader("Market Insights")
st.write(f"Reason: {insights['reason']}")
st.write(f"Suggestion: {insights['suggestion']}")

# Adding zoom functionality to plots
def plot_zoomable(df):
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines', name='100-day MA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df.index, y=ma200, mode='lines', name='200-day MA', line=dict(color='green')))

    fig.update_layout(
        title='Closing Price vs Time with Moving Averages',
        xaxis_title='Time',
        yaxis_title='Price',
        legend_title='Legend',
        hovermode='x unified'
    )

    return fig

st.plotly_chart(plot_zoomable(df))

# Adding doughnut chart for buy, sell, or hold recommendation
recommendation_labels = ['Buy', 'Sell', 'Hold']
recommendation_counts = [0, 0, 0]

if 'buy' in insights['suggestion'].lower():
    recommendation_counts[0] += 1
elif 'sell' in insights['suggestion'].lower():
    recommendation_counts[1] += 1
else:
    recommendation_counts[2] += 1

# Add sentiment-based recommendations
if positive_count > negative_count:
    recommendation_counts[0] += 1
elif negative_count > positive_count:
    recommendation_counts[1] += 1
else:
    recommendation_counts[2] += 1

# Plot recommendation doughnut chart
fig_recommendation = plt.figure(figsize=(3, 3))
plt.pie(recommendation_counts, labels=recommendation_labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#ff9999','#99ff99'], wedgeprops=dict(width=0.3))
plt.gca().set_facecolor('black')  # Setting black background
plt.axis('equal')
st.subheader("Recommendation: Buy, Sell, or Hold")
st.pyplot(fig_recommendation)
