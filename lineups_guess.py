import numpy as np
import pandas as pd

df = pd.read_csv("MLB/batter_all.csv", encoding='latin-1')
team_names = list(df['TEAM'].unique())

pitch = df[df['mlb_pos'] == 'P'][df['PA_batter_Std'] > 5]
pitch_mean = list(pitch.iloc[:,10:].mean())
pitch_mean[:10] = list(map((lambda x:int(x)),pitch_mean[:10]))
pitch_mean[34:48] = list(map((lambda x:int(x)),pitch_mean[34:48]))
pitch_mean[60:74] = list(map((lambda x:int(x)),pitch_mean[60:74]))
pp = ['P','P','P','P','P','P','P','P','P', 'P']
pitch_mean1 = pd.DataFrame([pp + pitch_mean], columns = pitch.columns)


team_lineup = {}
for team in team_names:
    if df[df['TEAM']==team][df['LG_batter_Std'] == 'AL'].empty:
        df_team = df[df['TEAM']==team].sort_values(['PA_batter_Std'], ascending = False)
        team_lineup[team] = pd.concat([df_team.iloc[0:9,:[2,4,6,7]], pitch_mean1.iloc[:,[2,4,6,7]]])
    elif df[df['TEAM']==team][df['LG_batter_Std'] == 'NL'].empty:
        df_team = df[df['TEAM']==team].sort_values(['PA_batter_Std'], ascending = False)
        team_lineup[team] = df_team.iloc[0:10,:]



team_lineup_right = {}
for team in team_names:
    df_team = df[df['TEAM']==team].sort_values(['PA_RIGHTbatter_Std'], ascending = False)
    team_lineup_right[team] = df_team.iloc[0:10,:]


team_lineup_left = {}
for team in team_names:
    df_team = df[df['TEAM']==team].sort_values(['PA_LEFT_batter_Std'], ascending = False)
    team_lineup_left[team] = df_team.iloc[0:10,:]




team_lineup.to_csv('MLB/team_lineup.csv', sep=',', encoding='utf-8')
team_lineup_right.to_csv('MLB/team_lineup_right.csv', sep=',', encoding='utf-8')
team_lineup_left.to_csv('MLB/team_lineup_left.csv', sep=',', encoding='utf-8')

