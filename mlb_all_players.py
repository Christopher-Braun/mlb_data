import numpy as np
import pandas as pd


master = pd.read_csv("MLB/mlb_all_players.csv", encoding='latin-1')

master = master.rename(columns={'mlb_team': 'TM'})
master['TM'] = list(map((lambda x:x.upper()),master['TM']))
master['TM'][master['TM']=='KC'] = 'KCA'
master['TM'][master['TM']=='SD'] = 'SDN'
master['TM'][master['TM']=='SF'] = 'SFN'
master['TM'][master['TM']=='WSH'] = 'WAS'
master['TM'][master['TM']=='TB'] = 'TBA'
master['TM'][master['TM']=='CHC'] = 'CHN'
master['TM'][master['TM']=='CWS'] = 'CHA'
master['TM'][master['TM']=='LAD'] = 'LAN'
master['TM'][master['TM']=='NYM'] = 'NYN'
master['TM'][master['TM']=='NYY'] = 'NYA'
master['TM'][master['TM']=='LAA'] = 'ANA'
master['TM'][master['TM']=='STL'] = 'SLN'


mlb_name = []
for name in master['mlb_name']:
    mlb_name.append(name.upper())

master['mlb_name'] = mlb_name

team_name = []
for name, team in zip(master['mlb_name'], master['TM']):
    team_name.append(name + ' ' + team)

master['NAME_TEAM'] = team_name

master.to_csv('MLB/mlb_all_players.csv', sep=',', encoding='utf-8')


