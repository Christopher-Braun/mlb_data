#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd


wiki = "https://legacy.baseballprospectus.com/sortable/index.php?cid=1918873"
page = urllib.request.urlopen(wiki)
soup = BeautifulSoup(page, 'lxml')

print(soup.prettify())

i=0
names = []
for i, row in enumerate(soup.find_all('a')):
    if (row.get_text()) == 'NAME':
        l = i
    elif (row.get_text()) == 'BWARP':
        le = i+1
    elif (row.get_text()) == 'Baseball Prospectus Home':
        li = i
    names.append(row.get_text())
    
categories = names[l:le]
player_team = names[le:li]

player, team = [], []
for i,j in enumerate(player_team):
    if i%2 == 0:
        player.append(j.upper())
    else:
        team.append(j)


stats, col, total = [], [], []
for row in soup.select('td[class="right"]'):
     stats.append(row.get_text())
     
        
i, j = 0, 0
for j in range(34,len(stats)+1,34):
    col=stats[i:j]
    total.append(col)
    col=[]
    i=j


league = soup.find_all(string=['AL', 'NL'])
league_list = list(league)

batter_team = pd.DataFrame(np.column_stack([player, team, league_list, total]), columns = categories)

batter_team_cols = list(batter_team.columns)

batter_team=batter_team[['NAME', 'TEAM', 'LG','PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR', 'TB', 'BB','SO','RBI', 'DP', 'SB', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BPF', 'oppOPS', 'TAv', 'VORP', 'FRAA', 'BWARP']]

batter_team_cols = list(batter_team.columns)

batter_team['PA']=batter_team['PA'].astype('int')
batter_team.loc[:,['PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR', 'TB', 'BB','SO','RBI', 'DP', 'SB','BPF']]=batter_team.loc[:,['PA', 'AB', 'R', 'H', '1B', '2B', '3B', 'HR', 'TB', 'BB','SO','RBI', 'DP', 'SB']].astype('int')
batter_team.loc[:,['AVG', 'OBP', 'SLG', 'OPS', 'ISO','oppOPS', 'TAv', 'VORP', 'FRAA', 'BWARP','BPF']]=batter_team.loc[:,['AVG', 'OBP', 'SLG', 'OPS', 'ISO','oppOPS', 'TAv', 'VORP', 'FRAA', 'BWARP','BPF']].astype('float')

batter_team['BASES_VS_AB'] = (batter_team['1B'] + batter_team['2B']*2 + batter_team['3B']*3 + batter_team['HR']*4 + batter_team['BB']) / batter_team['PA']

batter_team_cols = list(batter_team.columns)

de_cols = []
for col in batter_team_cols:
    if col != 'NAME' and col != 'TEAM' and col != 'NAME_TEAM':
        de_cols.append(col+'_batter_Std')
    else:
        de_cols.append(col)

batter_team.columns = de_cols

batter_team.to_csv('MLB/batter_team.csv', sep=',', encoding='utf-8')






















