#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd


wiki = "https://legacy.baseballprospectus.com/sortable/index.php?cid=1920411"
page = urllib.request.urlopen(wiki)
soup = BeautifulSoup(page, 'lxml')

print(soup.prettify())

i=0
names = []
for i, row in enumerate(soup.find_all('a')):
    if (row.get_text()) == 'NAME':
        l = i
    elif (row.get_text()) == 'TAv':
        le = i+1
    elif (row.get_text()) == 'Baseball Prospectus Home':
        li = i
    names.append(row.get_text().upper())
    
categories = names[l:le]
player_name = names[le:li]

stats, col, total = [], [], []
for row in soup.select('td[class="right"]'):
     stats.append(row.get_text())

for i,stat in enumerate(stats):
    if stat[-1] == '%':
        stats[i] = stat[:-1]     
        
i, j = 0, 0
for j in range(27,len(stats)+1,27):
    col=stats[i:j]
    total.append(col)
    col=[]
    i=j
    
categories.remove('HAND')
    
category_left = []
for category in categories:
    category = category + '_' + 'LEFT'
    category_left.append(category)
    
category_left[0] = 'NAME'

batter_vs_left = pd.DataFrame(np.column_stack([player_name, total]), columns = category_left)
batter_vs_left['G_LEFT']=batter_vs_left['G_LEFT'].astype('int')
batter_cols = list(batter_vs_left.columns)
batter_vs_left.loc[:,['G_LEFT', 'PA_LEFT', 'AB_LEFT', 'R_LEFT', 'H_LEFT', '1B_LEFT', '2B_LEFT', '3B_LEFT', 'HR_LEFT', 'TB_LEFT', 'SO_LEFT', 'BB_LEFT', 'HBP_LEFT', 'SF_LEFT', 'SH_LEFT', 'RBI_LEFT', 'DP_LEFT']]=batter_vs_left.loc[:,['G_LEFT', 'PA_LEFT', 'AB_LEFT', 'R_LEFT', 'H_LEFT', '1B_LEFT', '2B_LEFT', '3B_LEFT', 'HR_LEFT', 'TB_LEFT', 'SO_LEFT', 'BB_LEFT', 'HBP_LEFT', 'SF_LEFT', 'SH_LEFT', 'RBI_LEFT', 'DP_LEFT']].astype('int')
batter_vs_left.loc[:,['FB%_LEFT', 'GB%_LEFT', 'LD%_LEFT', 'POP%_LEFT', 'ISO_LEFT', 'AVG_LEFT', 'OBP_LEFT', 'SLG_LEFT', 'TAV_LEFT']]=batter_vs_left.loc[:,['FB%_LEFT', 'GB%_LEFT', 'LD%_LEFT', 'POP%_LEFT', 'ISO_LEFT', 'AVG_LEFT', 'OBP_LEFT', 'SLG_LEFT', 'TAV_LEFT']].astype('float')

batter_vs_left['BASES_VS_AB_LEFT'] = (batter_vs_left['1B_LEFT'] + batter_vs_left['2B_LEFT']*2 + batter_vs_left['3B_LEFT']*3 + batter_vs_left['HR_LEFT']*4 + batter_vs_left['BB_LEFT']) / batter_vs_left['PA_LEFT']
batter_vs_left = batter_vs_left.drop(columns = 'YEAR_LEFT')

batter_cols = list(batter_vs_left.columns)

de_cols = []
for col in batter_cols:
    if col != 'NAME' and col != 'TEAM' and col != 'NAME_TEAM':
        de_cols.append(col+'_batter_Std')
    else:
        de_cols.append(col)
        
batter_vs_left.columns = de_cols

batter_vs_left.to_csv('batter_vs_left.csv', sep=',', encoding='utf-8')
batter_vs_left.to_csv('mlb/batter_vs_left.csv', sep=',', encoding='utf-8')














