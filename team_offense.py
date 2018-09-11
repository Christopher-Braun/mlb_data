#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd



# Importing the dataset
team_bat = pd.read_csv("MLB/team_offense.csv", encoding='latin-1')
team_bat.info()

adv_team_bat = pd.read_csv("MLB/advanced_team_off.csv", encoding='latin-1')
team_list = pd.read_csv("MLB/team_list.csv", encoding='latin-1')

team_bat = team_bat.rename(columns={'Tm': 'TEAM'})
adv_team_bat = adv_team_bat.rename(columns={'Tm': 'TEAM'})

team_batting_stats = pd.merge(team_bat, adv_team_bat, on='TEAM', how='inner')

team_batting_stats_cols = list(team_batting_stats.columns)

team_batting_stats['1B'] = team_batting_stats['H'] - team_batting_stats['2B'] - team_batting_stats['3B'] - team_batting_stats['HR']
team_batting_stats['BASES_VS_AB'] = (team_batting_stats['1B'] + team_batting_stats['2B']*2 + team_batting_stats['3B']*3 + team_batting_stats['HR']*4 + team_batting_stats['BB']) / team_batting_stats['PA']
team_batting_stats = team_batting_stats.drop(columns = ['#Bat', 'BatAge'])

team_batting_stats_cols = list(team_batting_stats.columns)

de_cols = []
for col in team_batting_stats_cols:
    if col != 'NAME' and col != 'TEAM' and col != 'NAME_TEAM':
        de_cols.append(col+'_team_bat')
    else:
        de_cols.append(col)

team_batting_stats.columns = de_cols

#master = master.rename(columns={'mlb_team': 'TM'})
#master['TM'] = list(map((lambda x:x.upper()),master['TM']))
team_batting_stats['TEAM'][team_batting_stats['TEAM']=='KCR'] = 'KCA'
team_batting_stats['TEAM'][team_batting_stats['TEAM']=='SDP'] = 'SDN'
team_batting_stats['TEAM'][team_batting_stats['TEAM']=='SFG'] = 'SFN'
team_batting_stats['TEAM'][team_batting_stats['TEAM']=='WSN'] = 'WAS'
team_batting_stats['TEAM'][team_batting_stats['TEAM']=='TBR'] = 'TBA'
team_batting_stats['TEAM'][team_batting_stats['TEAM']=='CHC'] = 'CHN'
team_batting_stats['TEAM'][team_batting_stats['TEAM']=='CHW'] = 'CHA'
team_batting_stats['TEAM'][team_batting_stats['TEAM']=='LAD'] = 'LAN'
team_batting_stats['TEAM'][team_batting_stats['TEAM']=='NYM'] = 'NYN'
team_batting_stats['TEAM'][team_batting_stats['TEAM']=='NYY'] = 'NYA'
team_batting_stats['TEAM'][team_batting_stats['TEAM']=='LAA'] = 'ANA'
team_batting_stats['TEAM'][team_batting_stats['TEAM']=='STL'] = 'SLN'

team_batting_stats.to_csv('MLB/team_batting_stats.csv', sep=',', encoding='utf-8')










