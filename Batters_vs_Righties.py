#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


wiki = "https://legacy.baseballprospectus.com/sortable/index.php?cid=1920412"
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
    
category_right = []
for category in categories:
    category = category + '_' + 'RIGHT'
    category_right.append(category)
    
category_right[0] = 'NAME'

batter_vs_right = pd.DataFrame(np.column_stack([player_name, total]), columns = category_right)

batter_cols = list(batter_vs_right.columns)
batter_vs_right['G_RIGHT']=batter_vs_right['G_RIGHT'].astype('int')
batter_vs_right.loc[:,['G_RIGHT', 'PA_RIGHT', 'AB_RIGHT', 'R_RIGHT', 'H_RIGHT', '1B_RIGHT', '2B_RIGHT', '3B_RIGHT', 'HR_RIGHT', 'TB_RIGHT', 'SO_RIGHT', 'BB_RIGHT', 'HBP_RIGHT', 'SF_RIGHT', 'SH_RIGHT', 'RBI_RIGHT', 'DP_RIGHT']]=batter_vs_right.loc[:,['G_RIGHT', 'PA_RIGHT', 'AB_RIGHT', 'R_RIGHT', 'H_RIGHT', '1B_RIGHT', '2B_RIGHT', '3B_RIGHT', 'HR_RIGHT', 'TB_RIGHT', 'SO_RIGHT', 'BB_RIGHT', 'HBP_RIGHT', 'SF_RIGHT', 'SH_RIGHT', 'RBI_RIGHT', 'DP_RIGHT']].astype('int')
batter_vs_right.loc[:,['FB%_RIGHT', 'GB%_RIGHT', 'LD%_RIGHT', 'POP%_RIGHT', 'ISO_RIGHT', 'AVG_RIGHT', 'OBP_RIGHT', 'SLG_RIGHT', 'TAV_RIGHT']]=batter_vs_right.loc[:,['FB%_RIGHT', 'GB%_RIGHT', 'LD%_RIGHT', 'POP%_RIGHT', 'ISO_RIGHT', 'AVG_RIGHT', 'OBP_RIGHT', 'SLG_RIGHT', 'TAV_RIGHT']].astype('float')

batter_vs_right['BASES_VS_AB_RIGHT'] = (batter_vs_right['1B_RIGHT'] + batter_vs_right['2B_RIGHT']*2 + batter_vs_right['3B_RIGHT']*3 + batter_vs_right['HR_RIGHT']*4 + batter_vs_right['BB_RIGHT']) / batter_vs_right['PA_RIGHT']
batter_vs_right = batter_vs_right.drop(columns = 'YEAR_RIGHT')
batter_cols = list(batter_vs_right.columns)

de_cols = []
for col in batter_cols:
    if col != 'NAME' and col != 'TEAM' and col != 'NAME_TEAM':
        de_cols.append(col+'batter_Std')
    else:
        de_cols.append(col)
        
batter_vs_right.columns = de_cols


batter_vs_right.to_csv('MLB/batter_vs_right.csv', sep=',', encoding='utf-8')


    
    
    
