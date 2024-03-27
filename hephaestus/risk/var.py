import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from scipy.stats import norm
import pandas as pd
import json
from io import StringIO

def ValueatRisk(returns: pd.DataFrame, weights: np.ndarray, confidence: float = 0.95) -> float:
    covariance_matrix = returns.cov()
    average_returns = returns.mean()

    portfolio_mean = average_returns @ weights
    portfolio_std = np.sqrt(weights.T @ covariance_matrix @ weights)

    confidence_level = 1 - confidence
    
    return norm.ppf(confidence_level, portfolio_mean, portfolio_std)



def ReturnsNormDist(returns: pd.DataFrame, graph=False):
    pass
    

if __name__=='__main__':
    tickers = ['MS', 'TSLA', 'NVDA', 'PFE', 'F', 'MSFT', 'JPM', 'T', 'INTC', 'BAC']
    weights = np.array([0.1, 0.1, 0.2, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1])

    start = dt.datetime(2021, 1, 1)
    end = dt.datetime(2024, 1, 1)

    #data: pd.DataFrame = yf.download(tickers, start, end)['Adj Close']
    
    #json_data = data.to_json(orient='values')

    with open('data.json','r') as file:
        json_data = json.load(file)
        data = pd.read_json(StringIO(json_data))

    returns = data.pct_change()
    cov_matrix = returns.cov()
    avg_returns = returns.mean()
    count = returns.count().iloc(0)


    port_mean = avg_returns @ weights
    port_std = np.sqrt(weights.T @ cov_matrix @ weights)

    x = np.arange(-0.05, 0.05, 0.001)
    norm_dist = norm.pdf(x, port_mean, port_std)

    #plt.plot(x, norm_dist, color = 'g')
    #plt.show()

    confidence_level = 0.05

    VaR = norm.ppf(confidence_level, port_mean, port_std)
    print(VaR)

    print(ValueatRisk(returns, weights))




