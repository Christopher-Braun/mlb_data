#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

"""Individual Pitching"""
wiki = "https://legacy.baseballprospectus.com/sortable/index.php?cid=1905966"
page = urllib.request.urlopen(wiki)
soup = BeautifulSoup(page, 'lxml')

print(soup.prettify())

i=0
teams = []
for i, row in enumerate(soup.find_all('a')):
    if (row.get_text()) == 'TEAM':
        l = i
    elif (row.get_text()) == 'DP%':
        le = i+1
    elif (row.get_text()) == 'Baseball Prospectus Home':
        li = i
    teams.append(row.get_text())
    
categories = teams[l:le]
team = teams[le:li]


stats, col, total = [], [], []
for row in soup.select('td[class="right"]'):
     stats.append(row.get_text())
     
stats = list(map((lambda x:x.replace('1,', '1')), stats))
stats = list(map((lambda x:x.replace('2,', '2')), stats))
stats = list(map((lambda x:x.replace('3,', '3')), stats))
stats = list(map((lambda x:x.replace('4,', '4')), stats))
stats = list(map((lambda x:x.replace('5,', '5')), stats))
stats = list(map((lambda x:x.split('%')[0]), stats))
    
i, j = 0, 1
for j in range(18,len(stats)+17,18):
    col=stats[i+1:j]
    total.append(col)
    col=[]
    i=j


league = soup.find_all(string=['MLB'])
league_list = list(league)

defensive_effic = pd.DataFrame(np.column_stack([team, league_list, total]), columns = categories)
defensive_effic['PA']=defensive_effic['PA'].astype('int')
defensive_effic_cols = list(defensive_effic.columns)
defensive_effic.loc[:,['PA', 'AB', 'H', 'HR', 'SO', 'BB']]=defensive_effic.loc[:,['PA', 'AB', 'H', 'HR', 'SO', 'BB']].astype('int')
defensive_effic.loc[:,['DE', 'PADE', 'GB_DE', 'FB_DE', 'LINEDR_DE', 'POPUP_DE', 'GB%', 'FB%', 'LD%', 'POP%', 'DP%']]=defensive_effic.loc[:,['DE', 'PADE', 'GB_DE', 'FB_DE', 'LINEDR_DE', 'POPUP_DE', 'GB%', 'FB%', 'LD%', 'POP%', 'DP%']].astype('float')
defensive_effic = defensive_effic.drop(columns = 'LVL')
defensive_effic_cols = list(defensive_effic.columns)

de_cols = []
for col in defensive_effic_cols:
    if col != 'NAME' and col != 'TEAM' and col != 'NAME_TEAM':
        de_cols.append(col+'_Def_Ef')
    else:
        de_cols.append(col)
        
defensive_effic.columns = de_cols

defensive_effic.to_csv('MLB/defensive_effic.csv', sep=',', encoding='utf-8')


















