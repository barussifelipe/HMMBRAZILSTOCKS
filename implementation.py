import yfinance as yf
import functions as f 
from hmmlearn import hmm
import pandas as pd 


stocks = ["HBSA3.SA"]
#stocks = ["VALE3.SA", "PETR4.SA", "HBSA3.SA", "SAPR11.SA", "BBDC4.SA", "ITUB4.SA", "EUCA4.SA", "CMIG4.SA", "ELET3.SA"]
stocks = [f.subset_df(yf.download(stock, period="max")) for stock in stocks]

stock_model = hmm.GMMHMM(n_components = 3, n_mix = 3, covariance_type= "diag", random_state = 42, n_iter = 50)
#states = [buy, hold, sell]
#stock_model = stock_model.fit(stocks[0])
#z = stock_model.predict(stocks[0])
f.variables_plot(stocks[0])
