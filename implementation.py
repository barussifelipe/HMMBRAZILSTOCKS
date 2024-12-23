import yfinance as yf
import functions as f 
from hmmlearn import hmm
import pandas as pd 


stocks = ["HBSA3.SA"]
#stocks = ["VALE3.SA", "PETR4.SA", "HBSA3.SA", "SAPR11.SA", "BBDC4.SA", "ITUB4.SA", "EUCA4.SA", "CMIG4.SA", "ELET3.SA"]
stocks = [f.subset_df(yf.download(stock, period="max")) for stock in stocks]
len_df = stocks[0].shape[0]

stock_model = hmm.GMMHMM(n_components = 3, n_mix = 3, covariance_type= "diag", random_state = 42, n_iter = 50)
#states = [buy, hold, sell]

stock_model = stock_model.fit(stocks[0])

states = stock_model.predict(stocks[0])
states = pd.unique(states)

print(states)

z = stock_model.decode(stocks[0])[0]
pi = stock_model.startprob_

final_list = [pi[0]]
likelihood = f.likelihood_array(len_df, final_list, stocks, stock_model)

#Find the likelihood value, index, calculate the difference and then sum this diff to the adj closed value 
#Use matplotlib to plot the two values in the same plot. 
#Then we can define a function to find the prediction to any day that we want to compare efficiency.


    
         
            



#f.variables_plot(stocks_row=stocks[0].iloc[0])



        

    

 
