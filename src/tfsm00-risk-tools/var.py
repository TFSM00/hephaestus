import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from scipy.stats import norm
import pandas as pd

if __name__=='__main__':
    tickers = ['MS', 'TSLA', 'NVDA', 'PFE', 'F', 'MSFT', 'JPM', 'T', 'INTC', 'BAC']
    weights = np.array([0.1, 0.1, 0.2, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1])

    start = dt.datetime(2021, 1, 1)
    end = dt.datetime(2024, 1, 1)

    data: pd.DataFrame = yf.download(tickers, start, end)['Adj Close']

    returns = data.pct_change()
    cov_matrix = returns.cov()
    avg_returns = returns.mean()
    count = returns.count()[0]


    port_mean = avg_returns @ weights
    port_std = np.sqrt(weights.T @ cov_matrix @ weights)

    x = np.arange(-0.05, 0.05, 0.001)
    norm_dist = norm.pdf(x, port_mean, port_std)

    #plt.plot(x, norm_dist, color = 'g')
    #plt.show()

    confidence_level = 0.05

    VaR = norm.ppf(confidence_level, port_mean, port_std)

    print(returns)


