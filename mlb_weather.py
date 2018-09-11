#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

"""Individual Pitching"""
wiki = "http://dailybaseballdata.com/cgi-bin/weather.pl"
page = urllib.request.urlopen(wiki)
soup = BeautifulSoup(page, 'lxml')

print(soup.prettify())

date = []
for row in soup.select('td[valign="middle"]'):
    date.append(row.get_text().strip())

date1 = date[len(date)-1].split('for ')
date2 = date1[1]


temp = []
for row in soup.select('a[class="weather"]'):
    temp.append(row.get_text().strip())


table = []
for row in soup.select('span[style="background-color:white; "]'):
    table.append(row.get_text().strip())


temps = []
for row in soup.select('td[bgcolor="#FFFFCC"]'):
    temps.append(row.get_text().strip())

i=0
temp0, temp1, temp2 = [], [], []
for i in range(len(temps)):
    if i%3 == 1:
        temp1.append(temps[i])
    elif i%3 == 0:
        temp2.append(temps[i])
    elif i%3 == 2:
        temp0.append(temps[i])

i,j = 0, 0       
game, games = [], []
for t in temp2:
    if i < 5 and len(t) >= 1 and t[-1] != '%' and t[-1] != '°':
        game.append(t)
        i+=1
    elif i < 5 and len(t) >= 1 and t[-1] == '%':
        game.append(t[:-1])
        i+=1
    elif i < 5 and len(t) >= 1 and t[-1] == '°':
        game.append(t[:-1])
        i+=1
    elif i < 5 and not t:
        i+=1
    else:
        game.append(t)
        games.append(game)
        game = []
        i=0

density = []
for t in table:
    density.append(t[-2:])

teams = []
for t in temp:
    team = t.split(' at')
    team1 = team[1].split(':')
    teams.append(team1[0][:-3].strip())

dsl = []
for i in range(len(teams)):
    dsl.append(date2)


games_day = pd.DataFrame(games, index = teams, columns = ['TEMP', 'HUMIDITY', 'FEELS_LIKE', 'CONDITION', 'PRECIP%','WIND'])
games_day['AIR_DENSITY'] = density
games_day['DATE'] = dsl
games_day = games_day.reset_index()
games_day = games_day.rename(columns={"index": "TEAM"})

#games_day.to_csv('C:/Users/mrcrb/source/repos/MLB/mlb_weather.csv', sep=',', encoding='utf-8')

df_weather = pd.read_csv("MLB/mlb_weather.csv", encoding='latin-1')


import csv
with open(r'MLB/mlb_weather.csv', 'a', newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    for i in games_day.get_values():
        if games_day.iloc[len(games_day)-1,0] == df_weather.iloc[len(df_weather)-1,0]:
            print('Weather already updated')
            break
        else:
            writer.writerow(i)


















