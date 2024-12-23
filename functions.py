import pandas as pd
from bs4 import BeautifulSoup
import os 
import re
import yfinance as yf
import matplotlib.pyplot as plt 


def scraper(soup):
    table = soup.find("table", class_="histo-results")
    df = pd.read_html(str(table))[0]
    return df

def subset_df(df):
        
        df = df[["Open", 'Adj Close', "High", "Low"]]
        return df 

def subset_df_especial(df):
        
        df = df[["Open", "Close", "High", "Low"]]
        return df 

def brav3():
    dir_base = r"C:\Users\benga\Documents\Documentos\Programming\PYTHON\HIDDEN MARKOV STOCK\STOCKS\RRRP3"

    def inicial_fusion(dir_base: str):
    
        contents = sorted(os.listdir(dir_base), reverse= True, key = lambda x : list(map(int, re.findall(r"\d+", x)))[0])
        partial_dfs = list()
        for files in contents: 
            with open(dir_base + f"\{files}", encoding="UTF-8") as file_soup:
                soup = BeautifulSoup(file_soup, "lxml")
                partial_dfs.append(scraper(soup))
            
        return pd.concat(partial_dfs)


    inicial_fusion = subset_df_especial(inicial_fusion(dir_base))
    final_fusion = subset_df_especial(yf.download("BRAV3.SA", period="max")).reset_index(drop=True)
    BRAV3 = pd.concat([inicial_fusion, final_fusion], ignore_index=True)
    return BRAV3

def likelihood_array(len_df: int, final_list: list, stocks: list, stock_model):
    likelihood_array = [stock_model.decode(stocks[0].iloc[0:index])[0] for index in range(1, len_df+1)]
    final_list.append(likelihood_array)
    return final_list

def variables_plot(stocks_row):
    figure = plt.figure(figsize= (10, 10), layout = "constrained")
    axes = figure.add_subplot(2, 1, 1)
    axes.plot(stocks_row)
    axes.grid(True)
    axes.set_xlabel("Variables")
    axes.set_ylabel("Prices(R$)")
    figure.show()









