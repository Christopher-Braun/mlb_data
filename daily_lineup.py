#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

"""Individual Pitching"""
wiki = "https://www.rotowire.com/baseball/daily_lineups.htm"
page = urllib.request.urlopen(wiki)
soup = BeautifulSoup(page, 'lxml')

print(soup.prettify())

date = []
for row in soup.select('h1[class="dlineups-heading"]'):
     date.append(row.get_text())

ds = list(map((lambda x:x.split('MLB Daily Lineups for ')[1]),date))

for row in soup.find_all('a'):
    print(row.get_text())

i=0
games, li = [], []
for i, row in enumerate(soup.find_all('a')):
    if (row.get_text()) == 'Forecast.io':
        l = i+1
    elif (row.get_text()) == 'Advanced Search':
        le = i-3
    games.append(row.get_text())

games = games[l:le]

count = games.count('Enter your fantasy baseball lineup on FanDuel')
for i in range(count):
    games.remove('Enter your fantasy baseball lineup on FanDuel')


if 'Lineup Optimizer' in games:
    games.remove('Lineup Optimizer')
if 'Value Report' in games:    
    games.remove('Value Report')
if 'Matchups To Target/Avoid' in games:
    games.remove('Matchups To Target/Avoid')
if 'Odds Report' in games:
    games.remove('Odds Report')
if 'Daily Lineups App' in games:
    games.remove('Daily Lineups App')


for row in soup.find_all('p'):
    print(row.get_text())

weather = []
for row in soup.select('div[class="dlineups-topboxcenter-bottomline"]'):
     weather.append(row.get_text().strip())

forcast = []     
for days in weather:
    day = days.split()
    len(day)
    for i in range(len(day)):
        day[i] = day[i].strip()
    forcast.append(day)
        
ints, ints_not = [], []
for i,game in enumerate(games):
    if game[0].isdigit() and not games[i-1][0].isdigit() and not games[i+1][0].isdigit():
        ints_not.append(i)
        ints.append(i)
    elif game[0].isdigit() and not games[i+1][0].isdigit():
        ints.append(i)
    elif game[0].isdigit() and not games[i-1][0].isdigit():
        ints_not.append(i)
    elif i == len(games)-1:
        ints.append(i+1)
        ints_not.append(i+1)


i, j = 0, 0
time, home, away, pitch_h, pitch_a = [], [], [], [], []
for j in range(1,len(ints)):
    for i, game in enumerate(games):
        if ints[j] - ints[j-1] == 5:
            if i == ints[j-1]:
                time.append(game)
            elif i == ints[j-1] + 1:
                home.append(game)
            elif i == ints[j-1] + 2:
                away.append(game)
            elif i == ints_not[j-1] + 3:
                pitch_h.append(game)
            elif i == ints_not[j-1] + 4:
                pitch_a.append(game)
        else:    
            if i == ints[j-1]:
                time.append(game)
            elif i == ints[j-1] + 1:
                home.append(game)
            elif i == ints[j-1] + 2:
                away.append(game)
            elif i == ints_not[j] - 3:
                pitch_h.append(game)
            elif i == ints_not[j] - 2:
                pitch_a.append(game)

forcast2 = forcast.copy()

j = 0
for int_s, int_not in zip(ints, ints_not):
    if int_s != int_not:
        dif = int_s-int_not
        del forcast[j:j+dif]
        j += 1
    else:
        j+=1
        
mph, rain, dome = [], [], []
for wind in forcast:
    if wind[0] == 'Wind':
        if wind[3] == 'In':
            mph.append(int(wind[1])*int(-1))
        elif wind[3] == 'Out':
            mph.append(int(wind[1]))
        else:
            mph.append(int(0))
        rain.append(0)
        dome.append(0)
    if wind[0][0].isdigit():
        rain.append(int(wind[0][:-1]))
        mph.append(int(0))
        dome.append(0)
    if wind[0] == 'In':
        dome.append(1)
        rain.append(0)
        mph.append(int(0))
        
        
pitchers_add = []
for i,j in zip(pitch_h, pitch_a):
       pitchers_add.append(i)
       pitchers_add.append(j)
       
teams_add = []
for i,j in zip(home, away):
       teams_add.append(i)
       teams_add.append(j)
        
cols = [time, home, away, pitch_h, pitch_a, mph, rain, dome]
dsl = []
for i in range(len(home)):
    dsl.append(ds)

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
month_num = dict()
for month, day in zip(months, days):
    month_num[month] = day
ds1 = ds[0].split()
if len(ds1[1][:-1]) <= 1:
    ds1[1] = '0' + ds1[1][:-1]
date = ds1[2] + '-' + str(month_num[ds1[0]]) + '-' + ds1[1]
dates = []
for i in range(len(teams_add)):
    dates.append(date)

day_games = pd.DataFrame(np.column_stack([dates[:len(time)], time, home, away, pitch_h, pitch_a, mph, rain, dome]), columns = ['DATE','TIME', 'HOME', 'AWAY', 'PITCH_H', 'PITCH_A', 'MPH', 'RAIN', 'DOME'])
day_games.to_csv('MLB/day_games.csv', sep=',', encoding='utf-8')

# NEEDS WORK
day_pitchers = pd.DataFrame(np.column_stack([dates, pitchers_add, teams_add]), columns = ['DATE', 'PITCHER', 'TEAM'])

df = pd.read_csv("MLB/starting_pitchers.csv", encoding='latin-1')
df_w = pd.read_csv("MLB/day_weather.csv", encoding='latin-1')



import csv
fields=[pitch for pitch in day_pitchers[:]]
with open(r'MLB/starting_pitchers.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    for i in day_pitchers.get_values():
        if day_pitchers.iloc[len(day_pitchers)-1,0] == df.iloc[len(df)-1,0]:
            print('Players List already updated')
            break
        else:
            writer.writerow(i)


with open(r'MLB/day_weather.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    for i in day_games.get_values():
        if day_games.iloc[len(day_games)-1,0] == df_w.iloc[len(df_w)-1,0]:
            print('Weather already updated')
            break
        else:
            writer.writerow(i)



