#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

"""Team Batting"""
wiki = "https://legacy.baseballprospectus.com/sortable/index.php?cid=1918735"
page = urllib.request.urlopen(wiki)
soup = BeautifulSoup(page, 'lxml')

print(soup.prettify())

team = []
for i, row in enumerate(soup.find_all('a')):
    if (row.get_text()) == 'TEAM':
        l = i
    elif (row.get_text()) == 'VORP':
        le = i+1
    elif (row.get_text()) == 'Baseball Prospectus Home':
        li = i
    team.append(row.get_text())
    
categories = team[l:le]
team = team[le:li]


stats, col, total = [], [], []
for row in soup.select('td[class="right"]'):
     stats.append(row.get_text())
     
stats = list(map((lambda x:x.replace('1,', '1')), stats))
stats = list(map((lambda x:x.split('%')[0]), stats))
            
i=0
for j in range(28,len(stats)+1,28):
    col=stats[i:j]
    total.append(col)
    col=[]
    i=j


league = soup.find_all(string=['AL', 'NL'])
league_list = list(league)


batting = pd.DataFrame(np.column_stack([team, league_list, total]), columns = categories)
batting_cols = list(batting.columns)

batting['G']=batting['G'].astype('int')
batting.loc[:,['PA', 'AB', 'R', 'H', 'HR', 'TB', 'BB', 'IBB', 'SO','HBP', 'SF', 'SH', 'RBI', 'SB', 'CS', 'BPF']]=batting.loc[:,['PA', 'AB', 'R', 'H', 'HR', 'TB', 'BB', 'IBB', 'SO','HBP', 'SF', 'SH', 'RBI', 'SB', 'CS', 'BPF']].astype('int')
batting.loc[:,['BBr', 'SOr', 'SB', 'CS', 'SB%', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO','TAv', 'VORP']]=batting.loc[:,['BBr', 'SOr', 'SB', 'CS', 'SB%', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO','TAv', 'VORP']].astype('float')

batting.to_csv('MLB/batting_team.csv', sep=',', encoding='utf-8')

