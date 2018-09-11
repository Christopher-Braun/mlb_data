import numpy as np
import pandas as pd

df = pd.read_csv("MLB/mlb_master1.csv", encoding='latin-1')
df['Date'] = df['Date'].astype(str)
#df['Date'] = list(map((lambda x:x[:-2]),df['Date']))

game_ids = list(df['Game_ID'].unique())
#game_ids = game_ids[:-1]
team_ini = list(df['Team'].unique())
#team_ini = team_ini[:-1]
dates = list(df['Date'].unique())
#dates = dates[:-1]


df_home = df[df['H/A']=='h']

games_h, teams_h = [], []
for team in team_ini:
    for i, date in enumerate(dates):
        if date in df_home['Date'][df_home['Team']==team].values:
            games_h.append(df_home[df_home['Date']==date][df_home['Team']==team])
    teams_h.append(games_h)
    games_h = []

games, teams = [], []
for team in team_ini:
    for i, date in enumerate(dates):
        if date in df['Date'][df['Team']==team].values:
            games.append(df[df['Date']==date][df['Team']==team])
    teams.append(games)
    games = []


temps = []
for team in teams:
    for game in team:
        temps.append(list(game.iloc[0,:].values))

weather = pd.DataFrame(temps, columns=df_home.columns)

weather['W_dir'] = weather['W_dir'].astype(str)

wind_direction = []
for w in weather['W_dir']:
    if w[:3] == 'Out':
        wind_direction.append(1)
    elif w[:2] == 'In':
        wind_direction.append(-1)
    else:
        wind_direction.append(0)
        
weather['W_speed'] = weather['W_speed']*wind_direction

weather['Date'] = list(map((lambda x:x[:4] + '-' + x[4:6] + '-' + x[6:]), weather['Date']))
weather['Team'] = list(map((lambda x:x.upper()), weather['Team']))
weather['Oppt'] = list(map((lambda x:x.upper()), weather['Oppt']))


weather = weather.rename(columns={'Team': 'TEAM_OFFENSE'})
weather['TEAM_OFFENSE'] = list(map((lambda x:x.upper()),weather['TEAM_OFFENSE']))
weather['TEAM_OFFENSE'][weather['TEAM_OFFENSE']=='KAN'] = 'KCA'
weather['TEAM_OFFENSE'][weather['TEAM_OFFENSE']=='SDG'] = 'SDN'
weather['TEAM_OFFENSE'][weather['TEAM_OFFENSE']=='SFO'] = 'SFN'
weather['TEAM_OFFENSE'][weather['TEAM_OFFENSE']=='STL'] = 'SLN'
weather['TEAM_OFFENSE'][weather['TEAM_OFFENSE']=='TAM'] = 'TBA'
weather['TEAM_OFFENSE'][weather['TEAM_OFFENSE']=='CHC'] = 'CHN'
weather['TEAM_OFFENSE'][weather['TEAM_OFFENSE']=='CHW'] = 'CHA'
weather['TEAM_OFFENSE'][weather['TEAM_OFFENSE']=='LAD'] = 'LAN'
weather['TEAM_OFFENSE'][weather['TEAM_OFFENSE']=='LAA'] = 'ANA'
weather['TEAM_OFFENSE'][weather['TEAM_OFFENSE']=='NYM'] = 'NYN'
weather['TEAM_OFFENSE'][weather['TEAM_OFFENSE']=='NYY'] = 'NYA'

weather['Oppt'] = list(map((lambda x:x.upper()),weather['Oppt']))
weather['Oppt'][weather['Oppt']=='KAN'] = 'KCA'
weather['Oppt'][weather['Oppt']=='SDG'] = 'SDN'
weather['Oppt'][weather['Oppt']=='SFO'] = 'SFN'
weather['Oppt'][weather['Oppt']=='STL'] = 'SLN'
weather['Oppt'][weather['Oppt']=='TAM'] = 'TBA'
weather['Oppt'][weather['Oppt']=='CHC'] = 'CHN'
weather['Oppt'][weather['Oppt']=='CHW'] = 'CHA'
weather['Oppt'][weather['Oppt']=='LAD'] = 'LAN'
weather['Oppt'][weather['Oppt']=='LAA'] = 'ANA'
weather['Oppt'][weather['Oppt']=='NYM'] = 'NYN'
weather['Oppt'][weather['Oppt']=='NYY'] = 'NYA'

weather['Date_Team'] = weather['Date'] + ' ' + weather['TEAM_OFFENSE']
full_game_info = weather

weather_slim = weather[['Temp', 'W_speed', 'Date_Team']]
weather_slim = weather_slim.dropna()

weather_slim.to_csv('MLB/slim_temp_wind.csv', sep=',', encoding='utf-8')
weather.to_csv('MLB/mlb_temp_wind.csv', sep=',', encoding='utf-8')
        
