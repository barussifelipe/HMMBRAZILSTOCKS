import yfinance as yf
from functions import *
from hmmlearn import hmm
import pandas as pd
import math



# Download and process stock data
stocks_name = ["HBSA3.SA", "VALE3.SA", "BBDC4.SA", "PETR4.SA", "EUCA4.SA", "ELET3.SA", "POSI3.SA", "ITUB4.SA", "CMIG4.SA", "SAPR4.SA"]
stocks = [(stock, subset_df(yf.download(stock, period="max"))) for stock in stocks_name]
stocks.append(("BRAV3.SA", brav3()))
for stock in stocks:
    stock[1]["Returns"] = (stock[1]["Adj Close"] - stock[1]["Open"])/stock[1]["Open"]

    
    data = stock[1]
    training_data = stock[1].to_numpy()

    predicted_data = []
    actual_data = []

    stock_model = hmm.GaussianHMM(n_components=4, covariance_type="full",n_iter= 50, tol= 0.0001, random_state= 42, min_covar=1e-3)

    len_data = training_data.shape[0]
    current_data = training_data.copy()

    stock_model.fit(current_data)

    window_size = 30

    likelihood = likelihood_array(len_df= len_data, training_data = current_data, stock_model= stock_model, window_size= window_size)

    z = stock_model.score(current_data[-window_size:, :])
    matched_index = past_index(likelihood, z)

    difference = data.iloc[matched_index+1] - data.iloc[matched_index]

    actual_data = data.iloc[len_data-1]
    predicted_data = actual_data + difference

    print(f"For {stock[0]} the last closing price was", round(actual_data["Adj Close"], 2), "the predicted closing price is", round(predicted_data["Adj Close"], 2))


    
    
    
    









        

    

 
