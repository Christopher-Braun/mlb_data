#! python3
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
import pandas as pd

wiki = "https://legacy.baseballprospectus.com/sortable/index.php?cid=2565774"
page = urllib.request.urlopen(wiki)
soup = BeautifulSoup(page, 'lxml')

print(soup.prettify())

i=0
names = []
for i, row in enumerate(soup.find_all('a')):
    if (row.get_text()) == 'NAME':
        l = i
    elif (row.get_text()) == 'cFIP':
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


#[stat[:-1] for i,stat in enumerate(stats) if stat[-1] == '%']
        
i, j = 0, 0
for j in range(28,len(stats)+1,28):
    col=stats[i:j]
    total.append(col)
    col=[]
    i=j

hands = []    
for row in soup.select('td'):
    hands.append(row.get_text())
  
hands_ind = [i for i,hand in enumerate(hands) if hand == 'R' or hand == 'L']            
hands1 = [hands[i] for i in hands_ind]

categories.remove('HAND')
categories.insert(1,'HAND')

category_side = []
for category in categories:
    category = category + '_' + 'SIDE'
    category_side.append(category)
    
category_side[0] = 'NAME'
category_side[1] = 'HAND'
name_hand = pd.DataFrame(np.column_stack([player_name, hands1]), columns = categories[:2])

pitcher_vs = pd.DataFrame(total, columns = category_side[2:])
pitcher_vs_left = pd.concat([name_hand, pitcher_vs], axis=1)
pitcher_cols = list(pitcher_vs_left.columns)

pitcher_vs_left['G_SIDE']=pitcher_vs_left['G_SIDE'].astype('int')
pit_int = ['G_SIDE', 'PA_SIDE', 'AB_SIDE', 'H_SIDE', '1B_SIDE', '2B_SIDE', '3B_SIDE', 'HR_SIDE', 'TB_SIDE', 'SO_SIDE', 'BB_SIDE', 'HBP_SIDE', 'SF_SIDE', 'SH_SIDE', 'DP_SIDE']
pit_float = ['FB%_SIDE', 'GB%_SIDE', 'LD%_SIDE', 'POP%_SIDE', 'ISO_SIDE', 'AVG_SIDE', 'OBP_SIDE', 'SLG_SIDE', 'TAV_SIDE', 'DRA_SIDE', 'DRA-_SIDE', 'CFIP_SIDE']
for pit in pit_int:
    pitcher_vs_left[pit] = pitcher_vs_left[pit].astype(int)
for pit in pit_float:
    pitcher_vs_left[pit] = pitcher_vs_left[pit].astype(float)


pitcher_vs_left['BASES_VS_AB'] = (pitcher_vs_left['1B_SIDE'] + pitcher_vs_left['2B_SIDE']*2 + pitcher_vs_left['3B_SIDE']*3 + pitcher_vs_left['HR_SIDE']*4 + pitcher_vs_left['BB_SIDE']) / pitcher_vs_left['PA_SIDE']
pitcher_vs_left = pitcher_vs_left.drop(columns = 'YEAR_SIDE')

pitcher_cols = list(pitcher_vs_left.columns)

de_cols = []
for col in pitcher_cols:
    if col != 'NAME' and col != 'TEAM' and col != 'NAME_TEAM':
        de_cols.append(col+'_pit')
    else:
        de_cols.append(col)
        
pitcher_vs_left.columns = de_cols


pitcher_vs_left.to_csv('C:/Users/mrcrb/source/repos/MLB/pitcher_vs_left.csv', sep=',', encoding='utf-8')
pitcher_vs_left.to_csv('MLB/pitcher_vs_left.csv', sep=',', encoding='utf-8')
