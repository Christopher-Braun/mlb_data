#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Importing the dataset
player_data = pd.read_csv("MLB/player_ids.csv", encoding='latin-1', index_col=False)

for i,team in enumerate(player_data.iloc[:,2:]):
    for j,p in enumerate(player_data[team]):
        player_data.iloc[j,i+2] = p[1:-1]

names = []
for i,j in zip(player_data.iloc[:,11],player_data.iloc[:,12]):
    names.append(i.upper() + ' ' + j.upper())
    
player_data = player_data.rename(columns={"Last'": "'Last'"})

columns = list(player_data.columns)
col_new = []
for column in columns:
    col_new.append(column[1:-1])

player_data.columns = col_new
    
names_filt = []
for name in names:
    name1 = name.replace("\\'","'")
    names_filt.append(name1)
    
player_data['NAME'] = pd.Series(names_filt)

name_team = []
for i,j in zip(player_data.iloc[:,13],player_data.iloc[:,3]):
    name_team.append(i + ' ' + j)

player_data['NAME_TEAM'] = pd.Series(name_team)
player_data = player_data.iloc[:, [3,4,5,6,9,10,13,14]]

player_data.to_csv('MLB/player_data.csv', sep=',', encoding='utf-8')













