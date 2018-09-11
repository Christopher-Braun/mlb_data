#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

df = pd.read_csv("MLB/batter_all.csv", encoding='latin-1')

import lineups
lineups = lineups.lineups

day_lineups = pd.DataFrame(np.ones((1,len(df.columns))), columns = df.columns)
game_players = []
for line in lineups.iterrows():
    for name in line[1]:
        game_players.append(name)
        
names = list(df['NAME'].unique())
not_in = []
for name in names:
    if name not in game_players:
        not_in.append(name)

#ind = game_players.index('THOMAS PHAM')
#game_players[ind]='TOMMY PHAM'
#ind = game_players.index('DANIEL VOGELBACH')
#game_players[ind]='DAN VOGELBACH'
#ind = game_players.index('MIKE MARJAMA')
#game_players[ind]='MICHAEL MARJAMA'
#ind = game_players.index('OZZIE ALBIES')
#game_players[ind]='OZHAINO ALBIES'


lst_dict, not_inn = [], []
for player in game_players:
    val = df[df['NAME']==player].index.values
    if val.any():
        lst_dict.append(val)
    else:
        not_inn.append(player)

df_p = df[df['mlb_pos']=='P']
df_p = df_p.iloc[:,1:]
df_col = list(df_p.columns)
df_pslim = df_p.iloc[:, 9:]
average = []
for row,col in enumerate(df_pslim):
    average.append(np.mean(df_pslim[col]))
    
df = df.iloc[:,1:]

    
df2 = [99999, 'ANY', 'ALL', 'ANY', 'ALL', 'ANY', 'ANY', 'ALL', 'ALL']
df2 = df2 + average
df_len = len(df)
df.loc[df_len] = df2

i=int(len(df.index)+1)
for no in not_inn:
    df3 = df2
    df.loc[i]=df3
    i+=1

llst_dict= []
for player in game_players:
    val = df[df['NAME']==player].index.values
    if val.any():
        llst_dict.append(val[0])

batting_lineup = df.loc[llst_dict]
new_index = list(range(0,int(len(batting_lineup.index))))
batting_lineup = batting_lineup.reset_index()
batting_lineup = batting_lineup.iloc[:,1:]
batting_lineup1 = batting_lineup.iloc[:,9:]
#batting_lineup1=batting_lineup1.drop(columns='YEAR_RIGHT')
#batting_lineup1=batting_lineup1.drop(columns='YEAR_LEFT')


i,j = 0,0
team_lineup, lineup, mean, mean_group, teams = [],[],[],[],[]
for i in range(int(len(batting_lineup1.index))):
    if j < 9:
        lineup.append(list(batting_lineup1.loc[i]))
        j+=1
    else:
        mean_group.append(list(batting_lineup1.iloc[i-j:i,:].values))
        mean.append(list(batting_lineup1.iloc[i-j:i,:].mean()))
        team_lineup.append(lineup)
        teams.append(batting_lineup['TEAM'][i-j])
        lineup = []
        j=1

batting_lineup1_col = list(batting_lineup1.columns)    
bat_avg = pd.DataFrame(mean, index=teams, columns=batting_lineup1_col)

teams_not = []
team_list = pd.read_csv("MLB/team_list.csv", encoding='latin-1')
for team in team_list['TEAM']:
    if team not in list(bat_avg.index):
        teams_not.append(team)

# Change to put more weight on players with more plate appearances
df_non = df.iloc[:,6:]
team_mean = pd.DataFrame(np.ones((len(teams_not),len(df_non.columns))), index = teams_not, columns = df_non.columns)
for team in teams_not:
    team_mean.loc[team] = df_non[df_non['TEAM']==team].mean()
team_mean = team_mean.iloc[:, 2:]
bat_avg1 = pd.concat([bat_avg, team_mean])

bat_avg1.to_csv('MLB/batt_avg.csv', sep=',', encoding='utf-8')

