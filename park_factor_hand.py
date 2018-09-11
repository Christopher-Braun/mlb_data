#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

"""Individual Pitching"""
wiki = "https://legacy.baseballprospectus.com/sortable/index.php?cid=1819123"
page = urllib.request.urlopen(wiki)
soup = BeautifulSoup(page, 'lxml')

print(soup.prettify())

data = []
for row in soup.select('td[class="ctr"]'):
    data.append(row.get_text().strip())

data1 = []
for row in soup.select('a[target="_blank"]'):
    data1.append(row.get_text())


data2 = []
for row in soup.select('td[class="right"]'):
    data2.append(row.get_text())

i=0
stat, stats = [], []
for d in data2:
    if i < 14:
        stat.append(d)
        i+=1
    else:
        stat.append(d)
        stats.append(stat)
        stat = []
        i=0

data.remove('#')
data.remove('LVL')
data.remove('LG')
data.remove('TEAM')
    
park_factor_hand = pd.DataFrame(stats, columns = data )

park_factor_hand.insert(2,'TEAM',data1)

park_factor_hand=park_factor_hand.replace(to_replace='', value=0)

park_factor_hand['Home PA'] = park_factor_hand['Home PA'].astype(int)

park_factor_hand.iloc[:,3:] = park_factor_hand.iloc[:, 3:].astype(int)

park_factor_hand['FB Factor']/int(100)

for i in range(6,len(park_factor_hand.columns)):
    park_factor_hand.iloc[:, i] = park_factor_hand.iloc[:, i]/100

park_factor_hand.to_csv('MLB/park_factor_hand.csv', sep=',', encoding='utf-8')


