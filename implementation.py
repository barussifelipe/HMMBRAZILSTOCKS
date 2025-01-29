import yfinance as yf
from functions import *
from hmmlearn import hmm
import pandas as pd




# Download and process stock data
stocks = ["HBSA3.SA"]
stocks = [subset_df(yf.download(stock, period="max")) for stock in stocks]
stocks[0]["Returns"] = (stocks[0]["Adj Close"] - stocks[0]["Open"])/stocks[0]["Open"]
len_df = stocks[0].shape[0]

# Initialize and fit the model
stock_model = hmm.GMMHMM(n_components= 4, n_mix=3, covariance_type="diag", n_iter= 50, init_params="mcw")
stock_model.startprob_ = np.array([0.25, 0.25, 0.25, 0.25])
stock_model.transmat_ = np.array([
    [0.60, 0.20, 0.15, 0.05],  # Bullish -> Bullish, Bearish, Neutral, Volatile
    [0.20, 0.50, 0.20, 0.10],  # Bearish -> Bullish, Bearish, Neutral, Volatile
    [0.25, 0.25, 0.40, 0.10],  # Neutral -> Bullish, Bearish, Neutral, Volatile
    [0.15, 0.15, 0.30, 0.40]   # Volatile -> Bullish, Bearish, Neutral, Volatile
])


stock_model = stock_model.fit(stocks[0])



# Extract model states and probabilities
states = stock_model.predict(stocks[0])
states = pd.unique(states)
print("States:", states)

window_size = 60
current_window = stocks[0].iloc[len_df - window_size:len_df].to_numpy()

z = stock_model.score(current_window)
pi = stock_model.startprob_
print(z)
# Calculate likelihood array
initial_element = [pi[0]]

#likelihood = likelihood_array(len_df= len_df, initial_element= initial_element, stocks = stocks, stock_model= stock_model, window_size= window_size)
#matched_index = [index for index in range(0, len(likelihood)) if (z - (z*0.1) <= float(likelihood[index]) <= z + (z*0.01))]
#print(matched_index)
#best_index = find_most_representative_index(current_window, matched_index, stocks, window_size)

actual_data = stocks[0].iloc[len_df-1]
#past_data = stocks[0].iloc[best_index]

#Find the likelihood value, index, calculate the difference and then sum this diff to the adj closed value 
#Use matplotlib to plot the two values in the same plot. 
#Then we can define a function to find the prediction to any day that we want to compare efficiency.

#variables_plot(actual_data, past_data)



        

    

 
