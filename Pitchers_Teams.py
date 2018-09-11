#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

"""Individual Pitching"""
wiki = "https://legacy.baseballprospectus.com/sortable/index.php?cid=1931167"
page = urllib.request.urlopen(wiki)
soup = BeautifulSoup(page, 'lxml')

print(soup.prettify())

i=0
names = []
for i, row in enumerate(soup.find_all('a')):
    if (row.get_text()) == 'NAME':
        l = i
    elif (row.get_text()) == 'PWARP':
        le = i+1
    elif (row.get_text()) == 'Baseball Prospectus Home':
        li = i
    names.append(row.get_text())
    
categories = names[l:le]
player_team = names[le:li]

# Create seperate player and team lists
player, team = [], []
for i,j in enumerate(player_team):
    if i%2 == 0:
        player.append(j.upper())
    else:
        team.append(j)

# List of all stats
stats, col, total = [], [], []
for row in soup.select('td[class="right"]'):
     stats.append(row.get_text())
     
# Seperate stats by player        
i, j = 0, 0
for j in range(36,len(stats)+1,36):
    col=stats[i:j]
    total.append(col)
    col=[]
    i=j

league = soup.find_all(string=['AL', 'NL'])
league_list = list(league)

first, last = [], []
for play in player:
    sp = play.split()
    first.append(sp[0])
    last.append(sp[1:])
    
pitching = pd.DataFrame(np.column_stack([player, team, league_list, total]), columns = categories)

name_team = []
for i,j in zip(pitching.iloc[:,0],pitching.iloc[:,1]):
    name_team.append(i + ' ' + j)

pitching['NAME_TEAM'] = pd.Series(name_team)
pitching=pitching[['NAME', 'TEAM', 'LG', 'G', 'IP','PA', 'AB', 'R', 'ER', 'H', '1B', '2B', '3B', 'HR', 'TB', 'BB','SO','DRA', 'ERA', 'PPF', 'VORP', 'FIP', 'PVORP', 'PWARP', 'NAME_TEAM']]

pitching_cols = list(pitching.columns)

pitching['PA']=pitching['PA'].astype('int')
pitching.loc[:,['G', 'AB', 'R', 'ER', 'H', '1B', '2B', '3B', 'HR', 'TB', 'BB','SO', 'PPF']]=pitching.loc[:,['G', 'AB', 'R', 'ER', 'H', '1B', '2B', '3B', 'HR', 'TB', 'BB','SO', 'PPF']].astype('int')
pitching.loc[:,['IP', 'DRA', 'ERA', 'VORP', 'FIP', 'PVORP', 'PWARP']]=pitching.loc[:,['IP', 'DRA', 'ERA', 'VORP', 'FIP', 'PVORP', 'PWARP']].astype('float')

pitching['BASES_VS_AB'] = (pitching['1B'] + pitching['2B']*2 + pitching['3B']*3 + pitching['HR']*4 + pitching['BB']) / pitching['PA']

pitching_cols = list(pitching.columns)

de_cols = []
for col in pitching_cols:
    if col != 'NAME' and col != 'TEAM' and col != 'NAME_TEAM':
        de_cols.append(col+'_Pitching_Std')
    else:
        de_cols.append(col)

pitching.columns = de_cols

pitching.to_csv('MLB/pitching_team.csv', sep=',', encoding='utf-8')














