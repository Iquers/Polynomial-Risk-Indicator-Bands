# %%
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd


def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)

    # Applying logarithmic transformation
    data['Log_High'] = np.log(data['High'])
    data['Log_Low'] = np.log(data['Low'])

    return data


data = fetch_stock_data("AAPL", "2008-01-01", "2023-11-27")
# print(data[['High', 'Log_High', 'Low', 'Log_Low']])


def find_local_highs_lows(data, window_size):
    # Find local highs and lows within the rolling window
    data['Local_High'] = data['Log_High'].rolling(
        window=window_size, min_periods=1).max()
    data['Local_Low'] = data['Log_Low'].rolling(
        window=window_size, min_periods=1).min()

    return data


data['Log_Close'] = np.log(data['Close'])
window_size = 21  # number of days

data_with_locals = find_local_highs_lows(data, window_size)
# print(data_with_locals[['Log_High', 'Local_High', 'Log_Low', 'Local_Low']])


def plot_stock_with_curves(data, degree=3):
    # Convert date index to numeric values for curve fitting
    x = mdates.date2num(data.index.to_pydatetime())

    # Polynomial fitting for local highs and lows
    high_fit = np.polyfit(x, data['Local_High'], degree)
    low_fit = np.polyfit(x, data['Local_Low'], degree)

    high_poly = np.poly1d(high_fit)
    low_poly = np.poly1d(low_fit)

    # Generate x values for the curve plots
    x_curve = np.linspace(x.min(), x.max(), len(data))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Log_Close'],
             label='Log Stock Price', color='blue')
    plt.plot(mdates.num2date(x_curve), high_poly(
        x_curve), label='High Curve', color='red')
    plt.plot(mdates.num2date(x_curve), low_poly(
        x_curve), label='Low Curve', color='green')
    plt.title('Log Stock Price with High and Low Curves')
    plt.xlabel('Date')
    plt.ylabel('Log Price')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(
        interval=252))  # Adjust interval as needed
    plt.gcf().autofmt_xdate()  # Rotate date labels
    plt.legend()
    plt.show()


plot_stock_with_curves(data_with_locals, degree=3)


def plot_extended_curves(data, degree=3, extension_days=30):
    # Convert date index to numeric values for curve fitting
    x = mdates.date2num(data.index.to_pydatetime())

    # Polynomial fitting for local highs and lows
    high_fit = np.polyfit(x, data['Local_High'], degree)
    low_fit = np.polyfit(x, data['Local_Low'], degree)

    high_poly = np.poly1d(high_fit)
    low_poly = np.poly1d(low_fit)

    # Generate x values for the current and future curve plots
    last_date = data.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=extension_days)
    x_future = mdates.date2num(future_dates.to_pydatetime())

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(data.index, data['Log_Close'],
             label='Log Stock Price', color='blue')
    plt.plot(data.index, high_poly(x), label='High Curve', color='red')
    plt.plot(data.index, low_poly(x), label='Low Curve', color='green')
    # Plotting extended curves
    plt.plot(future_dates, high_poly(x_future),
             label='Extended High Curve', color='lightcoral', linestyle='--')
    plt.plot(future_dates, low_poly(x_future),
             label='Extended Low Curve', color='lightgreen', linestyle='--')

    plt.title('Log Stock Price with Extended High and Low Curves')
    plt.xlabel('Date')
    plt.ylabel('Log Price')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(
        mdates.MonthLocator(interval=12))  # Adjust as needed
    plt.gcf().autofmt_xdate()  # Rotate date labels
    plt.legend()
    plt.show()


plot_extended_curves(data_with_locals, degree=3, extension_days=504)
