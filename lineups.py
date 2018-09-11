#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

"""Individual Pitching"""
wiki = "https://rotogrinders.com/lineups/mlb?site=draftkings"
page = urllib.request.urlopen(wiki)
soup = BeautifulSoup(page, 'lxml')

print(soup.prettify())

teams = []    
for row in soup.select('span[class="shrt"]'):
    teams.append(row.get_text())

play1 = []    
for row in soup.select('span[class="pname"]'):
    play1.append(row.get_text())
    
player_info = []
for pla in play1:
    player_info.append(pla.strip("'\n'").upper())


i,j = 0,0
team_lineup, lineup = [],[]    
for i in range(len(player_info)):
    if j < 8:
        lineup.append(player_info[i])
        j+=1
    else:
        lineup.append(player_info[i])
        team_lineup.append(lineup)
        lineup = []
        j=0

lineups = pd.DataFrame(team_lineup, index = teams)

lineups.to_csv('MLB/lineups.csv', sep=',', encoding='utf-8')













