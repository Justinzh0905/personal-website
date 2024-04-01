from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import yfinance as yf
from datetime import date, timedelta
import statsmodels.api as sm
import plotly.express as px
import scipy.optimize as sco 
from django_plotly_dash import DjangoDash
from main.models import Stock

yf.pdr_override()

#use 1wk or 1mo
interval = '1wk'

if interval == '1mo':
    multiple = 12
else:
    multiple = 52
tickers = ['SPY'] + [stock.ticker for stock in Stock.objects.all()]

start = date.today() - timedelta(days=365*5)
data = {ticker: yf.download(ticker, start=start, interval=interval)['Adj Close'] for ticker in tickers}

data = pd.concat(data, axis=1).dropna().pct_change()

rf = pdr.DataReader('DGS10', 'fred', start)['DGS10']
rf = rf[rf.index.isin(data.index)] / 100
rf = np.log(1+rf)/52

data = pd.concat([rf, data], axis=1)
data.dropna(inplace=True)
data['SPY Excess'] = data['SPY'] - rf
betas = {}

for ticker in tickers:
    if ticker != 'SPY':
        data[ticker + ' Excess'] = data[ticker] - rf
    model = sm.OLS(data[ticker + ' Excess'], data["SPY Excess"]).fit()
    betas[ticker] = model.params.iloc[0]

betas = pd.Series(betas)

returns = {ticker: (betas[ticker] * data['SPY Excess'].mean() + rf.mean()) for ticker in tickers }

stocks = data[tickers[1:]]

def portfolio_annualised_performance(weights, expected_returns, cov_matrix):
    returns = (np.sum(expected_returns*weights ) + 1) ** multiple - 1
    std = np.sqrt(np.dot(weights.T @ cov_matrix, weights))[0] * np.sqrt(multiple)
    return std, returns

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[:,i] = portfolio_std_dev, portfolio_return, (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

def simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate, ratio='sharpe'):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    df = pd.DataFrame(results).transpose()
    df.columns = ['Standard Deviation', 'Expected Returns', f'{ratio} Ratio']
    return df

def neg_ratio(weights, returns, cov, rf):
    std, expected_return = portfolio_annualised_performance(weights, returns, cov)
    return -1* (expected_return - rf)/std

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(multiple)

def efficient_return(expected_returns, cov_matrix, target):
    num_assets = len(expected_returns)
    args = (cov_matrix)

    def portfolio_return(weights):
        return (np.sum(expected_returns*weights ) + 1) ** multiple - 1

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range, rf, ratio='sharpe'):
    efficients = []
    for ret in returns_range:
        trial = efficient_return(mean_returns, cov_matrix, ret)['fun']
        efficients.append(trial)
    df = pd.DataFrame(efficients)
    df = pd.concat([df, pd.Series(returns_range)], axis=1)
    df.columns = ['Standard Deviation', 'Expected Returns']
    df[f'{ratio} Ratio'] = (df['Expected Returns'] - rf) / df['Standard Deviation']
    return df.round(5)

returns = pd.Series(list(returns.values())[1:])
returns.index = tickers[1:]

def semivar(stock1, stock2, rf):
    sum = 0
    for i in stock1.index:
        sum += min(stock1[i] - rf[i], 0) * min(stock2[i] - rf[i], 0)
    return sum / len(stock1)

def semicovar(stocks, rf):
    return pd.DataFrame([[semivar(stocks[stock1], stocks[stock2], rf) for stock1 in stocks.columns] for stock2 in stocks.columns], index=stocks.columns, columns=stocks.columns)


app = DjangoDash('chart')

app.layout = html.Div([
    dcc.Dropdown(['Sharpe', 'Sortino'], 'Sharpe', id='ratio-select'),
    dcc.Graph(figure={}, id="frontier-graph")
], style={'margin-inline': '5%'})


graphing_mode = 'optimize'

@app.callback(
    Output(component_id='frontier-graph', component_property='figure'),
    Input(component_id='ratio-select', component_property='value')
)
def update_graph(ratio):
    if ratio == 'Sharpe':
        cov = stocks.cov()
    else:
        cov = semicovar(stocks, rf)
    
    df = None
    if graphing_mode =='optimize':
        returns_range = (1 + np.linspace(returns.min(), returns.max(), num=150))**multiple - 1
        df = efficient_frontier(returns, cov, returns_range, rf.mean(), ratio)
    elif graphing_mode =='random':
        df = simulated_ef_with_random(returns, cov, 3000, rf.mean(), ratio)
    fig = px.scatter(df, x='Standard Deviation', y='Expected Returns', color=f'{ratio} Ratio')

    max_ratio = df[f'{ratio} Ratio'].idxmax()
    fig.add_traces(
        px.scatter(df.loc[[max_ratio]], x='Standard Deviation', y='Expected Returns', hover_data=df.columns).update_traces(marker_size=20, marker_color="black").data
    )

    individual = []
    for ticker in tickers[1:]:
        if ticker:
            individual.append(list(portfolio_annualised_performance(np.array([1.]),np.array([returns[ticker]]), np.array([cov.loc[ticker, ticker]]))))
            individual[-1].append((individual[-1][1] - rf.mean())/individual[-1][0])
            individual[-1].insert(0, ticker)

    individual = pd.DataFrame(individual)
    individual = individual.round(5)
    individual.columns = ['Ticker', 'Standard Deviation', 'Expected Returns', f'{ratio} Ratio']

    fig.add_traces(
        px.scatter(individual, x='Standard Deviation', y='Expected Returns', hover_name='Ticker', hover_data=df.columns).update_traces(marker_size=10, marker_color="black").data
    )

    return fig

if __name__ == '__main__':
    app.run(debug=True)