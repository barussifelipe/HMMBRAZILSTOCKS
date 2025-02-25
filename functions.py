import pandas as pd
from bs4 import BeautifulSoup
import os 
import re
import yfinance as yf
import matplotlib.pyplot as plt 
import numpy as np
from hmmlearn import hmm


def scraper(soup):
    table = soup.find("table", class_="histo-results")
    df = pd.read_html(str(table))[0]
    return df

def subset_df(df):
    df = df[["Open", 'Adj Close', "High", "Low", "Volume"]]
    return df 

def subset_df_especial(df):
    df = df[["Open", "Close", "High", "Low"]]
    return df 

def inicial_fusion(dir_base: str):
    contents = sorted(os.listdir(dir_base), reverse= True, key = lambda x : list(map(int, re.findall(r"\d+", x)))[0])
    partial_dfs = list()
    for files in contents: 
        with open(dir_base + f"\{files}", encoding="UTF-8") as file_soup:
            soup = BeautifulSoup(file_soup, "lxml")
            partial_dfs.append(scraper(soup))
            
    return pd.concat(partial_dfs)

def brav3():
    dir_base = r"C:\Users\benga\Documents\Documentos\Programming\PYTHON\HIDDEN MARKOV STOCK\STOCKS\RRRP3"
    inicial_f = subset_df_especial(inicial_fusion(dir_base))
    final_fusion = subset_df_especial(yf.download("BRAV3.SA", period="max")).reset_index(drop=True)
    BRAV3 = pd.concat([inicial_f, final_fusion], ignore_index=True)
    return BRAV3


def past_index(likelihood_array: list, z):
    match_index = np.argmin([np.abs(x - z) for x in likelihood_array])
    return match_index

def likelihood_array(len_df: int, training_data: list, stock_model, window_size: int): 
    likelihood_array = [stock_model.score(training_data[index:index+window_size,:]) for index in range(1, len_df + 1 - (2*window_size))]
    return likelihood_array


def variables_plot(stocks_row1, stocks_row2):
    figure = plt.figure(figsize= (10, 10), layout = "constrained")
    axes = figure.add_subplot(2, 1, 1)
    line1, = axes.plot(stocks_row1)
    line2, = axes.plot(stocks_row2)
    line1.set_label("Actual Data")
    line2.set_label("Past Data")
    axes.legend()
    axes.grid(True)
    axes.set_xlabel("Variables")
    axes.set_ylabel("Prices(R$)")
    figure.show()







