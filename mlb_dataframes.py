#! python3    
import numpy as np
import pandas as pd
import Batters_Teams
import Pitchers_Teams
import Batters_vs_Lefties
import Batters_vs_Righties
import Pitchers_vs_Lefties
import Pitchers_vs_Righties
import mlb_defensive_effic
import mlb_standings
import mlb_game_scores
import mlb_player_info
import mlb_starting_pitchers
import team_offense
import lineups
import batter_all
import park_factor_hand
import master

batters = Batters_Teams.batter_team
pitchers = Pitchers_Teams.pitching
batters_left = Batters_vs_Lefties.batter_vs_left
batters_right = Batters_vs_Righties.batter_vs_right
pitchers_left = Pitchers_vs_Lefties.pitcher_vs_left
pitchers_right = Pitchers_vs_Righties.pitcher_vs_right
defensive_efficiency = mlb_defensive_effic.defensive_effic
standings_order = mlb_standings.standings
game_scores = mlb_game_scores.scoring
player_info = mlb_player_info.player_data
starting_pitchers = mlb_starting_pitchers.win_pitch
team_offense = team_offense.team_batting_stats
lineups = lineups.lineups
bat_avg = batter_all.bat_avg1
park_factor_hand = park_factor_hand.park_factor_hand
wind_temp = master.weather
slim_wind_temp = master.weather_slim

'''from PythonScripts.MLB import Batters_Teams
batters = Batters_Teams.batter_team

from PythonScripts.MLB import Pitchers_Teams
pitchers = Pitchers_Teams.pitching

from PythonScripts.MLB import Batters_vs_Lefties
batters_left = Batters_vs_Lefties.batter_vs_left

from PythonScripts.MLB import Batters_vs_Righties
batters_right = Batters_vs_Righties.batter_vs_right

from PythonScripts.MLB import Pitchers_vs_Lefties
pitchers_left = Pitchers_vs_Lefties.pitcher_vs_left

from PythonScripts.MLB import Pitchers_vs_Righties
pitchers_right = Pitchers_vs_Righties.pitcher_vs_right

from PythonScripts.MLB import mlb_defensive_effic
defensive_efficiency = mlb_defensive_effic.defensive_effic

from PythonScripts.MLB import mlb_standings
standings_order = mlb_standings.standings

from PythonScripts.MLB import mlb2
batting = mlb2.batting

from PythonScripts.MLB import mlb_game_scores
game_scores = mlb_game_scores.scoring

from PythonScripts.MLB import mlb_player_info
player_info = mlb_player_info.player_data

from PythonScripts.MLB import daily_lineup
day_games = daily_lineup.day_games

from PythonScripts.MLB import daily_lineup
day_weather = daily_lineup.df_w

from PythonScripts.MLB import mlb_starting_pitchers
starting_pitchers = mlb_starting_pitchers.win_pitch

from PythonScripts.MLB import team_offense
team_offense = team_offense.team_batting_stats

from PythonScripts.MLB import lineups
lineups = lineups.lineups

from PythonScripts.MLB import batter_all
bat_avg = batter_all.bat_avg1

from PythonScripts.MLB import park_factor_hand
park_factor_hand = park_factor_hand.park_factor_hand

from PythonScripts.MLB import master
wind_temp = master.weather

from PythonScripts.MLB import master
slim_wind_temp = master.weather_slim'''

master = pd.read_csv("MLB/mlb_all_players.csv", encoding='latin-1')
game_scores.to_csv("C:/Users/mrcrb/PythonScripts/MLB/MLB/game_scores.csv", sep=',', encoding='utf-8', index=False)
'''
f = open(r'MLB\schedule.log', 'w')
f.write('Starting Script')
'''    
# Check if player is in both dataframes
def player_check(frame1, frame2, frame1_name, frame2_name):
    frame1_not = []
    for player in list(frame1[frame1_name]):
        if player not in list(frame2[frame2_name]):
            frame1_not.append(player)
    return frame1_not

# Combine pitchers with pitcher info
#pitcher_template = pd.merge(player_info, pitchers, on='NAME_TEAM', how='inner')
pitcher_template1 = pd.merge(master, pitchers, on='NAME_TEAM', how='inner')
#pitchers_not = player_check(pitchers,pitcher_template,'NAME','NAME_x')
pitchers_not1 = player_check(pitchers, pitcher_template1, 'NAME', 'mlb_name')
pitchers_template2 = pd.concat([pitcher_template1.iloc[:, 2:9], pitcher_template1.iloc[:, -27:]], axis=1)

#col_del = ['Team(s)','NAME_x', 'DRA_Pitching_Std', 'VORP_Pitching_Std','PVORP_Pitching_Std', 'PWARP_Pitching_Std']
#pitcher_template = pitcher_template.drop(col_del, axis=1)

# Batters Team Info, Batting Stats vs All Pitchers
batter_template = pd.merge(player_info, batters, on='NAME', how='inner')
batters['NAME_TEAM'] = batters['NAME'] + ' ' + batters['TEAM']
batter_template1 = pd.merge(master, batters, on='NAME_TEAM', how='inner')
batters_not1 = player_check(batters, batter_template1, 'NAME', 'mlb_name')
batters_not = player_check(batters, batter_template, 'NAME', 'NAME')

col_del = ['Throws', 'BPF_batter_Std']
col_del1 = ['throws', 'BPF_batter_Std']

batter_template = batter_template.drop(col_del, axis=1)
batter_template1 = batter_template1.drop(col_del1, axis=1)
batter_template2 = pd.concat([batter_template1.iloc[:, 2:8], batter_template1.iloc[:, 38:]], axis=1)


# Combine pitchers_template with pitchers_right
#pitcher_template = pitcher_template.rename(columns={'NAME_y': 'NAME'})
#pitcher_total = pd.merge(pitchers_template2, pitchers_right, on='NAME', how='inner')
#pitchers_not_t = player_check(pitchers_template2, pitcher_total, 'mlb_name', 'NAME')

# Combine pitchers_template with pitchers_left
#pitcher_total = pitcher_total.rename(columns={'NAME_x': 'NAME'})
pitcher_all = pd.merge(pitchers_template2, pitchers_left, on='NAME', how='inner')
pitchers_not_all = player_check(pitchers_template2, pitcher_all, 'NAME', 'NAME')

# Batters Team Info, Batting Stats vs All Pitchers and vs Right Handed Pitchers
batter_total = pd.merge(batter_template2, batters_right, on='NAME', how='inner')
batters_not_t = player_check(batter_template2, batter_total, 'mlb_name', 'NAME')

# Batters Team Info, Batting Stats vs All Pitchers, Right Handed and Left Handed Pitchers
batter_all = pd.merge(batter_total, batters_left, on='NAME', how='inner')

batters_not_all = player_check(batter_total, batter_all, 'NAME', 'NAME')

for t1, t2 in zip(batter_all['TM'], batter_all['TEAM']):
    if t1 != t2:
        print(t1, t2)
        
batter_all = batter_all.drop(batter_all[batter_all['TM'] != batter_all['TEAM']].index)
        
bat_col = list(batter_all.columns)
pit_col = list(pitcher_all.columns)
					
col_del1 = ['bats', 'NAME_TEAM', 'LG_Pitching_Std', 'AB_Pitching_Std', 'H_Pitching_Std', 'AB_SIDE_pit', 'H_SIDE_pit', 'HBP_SIDE_pit']
pitcher_all = pitcher_all.drop(col_del1, axis=1)
pit_col = list(pitcher_all.columns)


col_del2 = ['TM', 'NAME_TEAM', 'AB_batter_Std', 'AB_RIGHTbatter_Std', 'AB_LEFT_batter_Std']
batter_all = batter_all.drop(col_del2, axis=1)
bat_col = list(batter_all.columns)

sdthb_p = []
for i in range(0, len(pitcher_all['mlb_id'])):
    x = (pitcher_all.loc[i, '1B_SIDE_pit'] + pitcher_all.loc[i, '2B_SIDE_pit']*int(2) + pitcher_all.loc[i, '3B_SIDE_pit']*int(3) + pitcher_all.loc[i, 'HR_SIDE_pit']*int(4) + pitcher_all.loc[i, 'BB_SIDE_pit'])/pitcher_all.loc[i, 'PA_SIDE_pit']
    sdthb_p.append(x)

pitcher_all['SDTHB_P'] = sdthb_p

'''
sdthb_r = []
for i in range(0, len(pitcher_all['mlb_id'])):
    x = (pitcher_all.loc[i, '1B_RIGHT_pitcher_Std'] + pitcher_all.loc[i, '2B_RIGHT_pitcher_Std']*int(2) + pitcher_all.loc[i, '3B_RIGHT_pitcher_Std']*int(3) + pitcher_all.loc[i, 'HR_RIGHT_pitcher_Std']*int(4) + pitcher_all.loc[i, 'BB_RIGHT_pitcher_Std'])/pitcher_all.loc[i, 'PA_RIGHT_pitcher_Std']
    sdthb_r.append(x)

pitcher_all['SDTHB_R'] = sdthb_r

sdthb_l = []
for i in range(0, len(pitcher_all['mlb_id'])):
    xr = (pitcher_all.loc[i, '1B_LEFT_pitcher_Std'] + pitcher_all.loc[i, '2B_LEFT_pitcher_Std']*int(2) + pitcher_all.loc[i, '3B_LEFT_pitcher_Std']*int(3) + pitcher_all.loc[i, 'HR_LEFT_pitcher_Std']*int(4) + pitcher_all.loc[i, 'BB_LEFT_pitcher_Std'])/pitcher_all.loc[i, 'PA_LEFT_pitcher_Std']
    sdthb_l.append(xr)

pitcher_all['SDTHB_L'] = sdthb_l
'''

sdthb_all = []
for i in range(0, len(pitcher_all['mlb_id'])):
    xl = (pitcher_all.loc[i, '1B_Pitching_Std'] + pitcher_all.loc[i, '2B_Pitching_Std']*int(2) + pitcher_all.loc[i, '3B_Pitching_Std']*int(3) + pitcher_all.loc[i, 'HR_Pitching_Std']*int(4) + pitcher_all.loc[i, 'BB_Pitching_Std'])/pitcher_all.loc[i, 'PA_Pitching_Std']
    sdthb_all.append(xl)

pitcher_all['SDTHB_ALL'] = sdthb_all 
pitcher_all1 = pd.merge(pitcher_all, defensive_efficiency, on='TEAM', how='inner')
pitcher_games = list(pitcher_all1.columns)
pitcher_all1.to_csv('C:/Users/mrcrb/source/repos/MLB/pitcher_all.csv', sep=',', encoding='utf-8', columns=pitcher_games)
pitcher_all1.to_csv("C:/Users/mrcrb/PythonScripts/MLB/MLB/pitcher_all.csv", sep=',', encoding='utf-8', index=False)


game_scores1 = pitcher_all1.rename(columns={'NAME': 'Pitcher'})
game_scores = game_scores.rename(columns={'PITCHER_y': 'Pitcher'})

pitcher_games = pd.merge(game_scores, game_scores1, on='Pitcher', how='inner')

game_scores_not = player_check(game_scores, game_scores1, 'Pitcher', 'Pitcher')

batter_all_col = list(batter_all.columns)
batter_all = batter_all.drop(batter_all[batter_all['PA_batter_Std'] == 0].index)
batter_all = batter_all.reset_index()
batter_all = batter_all.iloc[:, 1:]

sdthb_bat = []
for i in range(0, len(batter_all['mlb_id'])):
    x = (batter_all.loc[i, '1B_batter_Std'] + batter_all.loc[i, '2B_batter_Std']*int(2) + batter_all.loc[i, '3B_batter_Std']*int(3) + batter_all.loc[i, 'HR_batter_Std']*int(4) + batter_all.loc[i, 'BB_batter_Std'])/batter_all.loc[i, 'PA_batter_Std']
    sdthb_bat.append(x)

batter_all['SDTHB_BAT'] = sdthb_bat

sdthb_batr = []
for i in range(0, len(batter_all['mlb_id'])):
    xr = (batter_all.loc[i, '1B_RIGHTbatter_Std'] + batter_all.loc[i, '2B_RIGHTbatter_Std']*int(2) + batter_all.loc[i, '3B_RIGHTbatter_Std']*int(3) + batter_all.loc[i, 'HR_RIGHTbatter_Std']*int(4) + batter_all.loc[i, 'BB_RIGHTbatter_Std'])/batter_all.loc[i, 'PA_RIGHTbatter_Std']
    sdthb_batr.append(xr)

batter_all['SDTHB_BATR'] = sdthb_batr

sdthb_batl = []
for i in range(0, len(batter_all['mlb_id'])):
    xl = (batter_all.loc[i, '1B_LEFT_batter_Std'] + batter_all.loc[i, '2B_LEFT_batter_Std']*int(2) + batter_all.loc[i, '3B_LEFT_batter_Std']*int(3) + batter_all.loc[i, 'HR_LEFT_batter_Std']*int(4) + batter_all.loc[i, 'BB_LEFT_batter_Std'])/batter_all.loc[i, 'PA_LEFT_batter_Std']
    sdthb_batl.append(xl)

batter_all['SDTHB_BATL'] = sdthb_batl
bat_games = list(batter_all.columns)
batter_all.to_csv('C:/Users/mrcrb/source/repos/MLB/batter_all.csv', sep=',', encoding='utf-8', columns=bat_games)
batter_all.to_csv("C:/Users/mrcrb/PythonScripts/MLB/MLB/batter_all.csv", sep=',', encoding='utf-8', index=False)

#game_bat = batter_all[bat_games] 
game_scores1 = game_scores1.rename(columns={'Pitcher': 'NAME'})

# Add Team Initials to to pitcher_games
#standings_order = standings_order.iloc[:-1,:]
#team_ini = list(standings_order['TM'].unique())
#team_long = list(standings_order['NAME'].unique())
#team_chart = pd.DataFrame(np.column_stack([team_long, team_ini]), columns=['Team', 'TEAM_OFFENSE'])

long = list(pitcher_games['mlb_team_long'].unique())
tm = list(pitcher_games['TM'].unique())
team_list = pd.DataFrame(np.column_stack([long, tm]), columns=['Team', 'TEAM'])
team_list.to_csv('C:/Users/mrcrb/source/repos/MLB/team_list.csv', sep=',', encoding='utf-8', index=False)
team_list['Team'][team_list['Team'] == 'Arizona Diamondbacks'] = "Arizona D'Backs"
pitcher_games = pitcher_games.rename(columns={'TEAM_y':'TEAM'})
pitcher_games = pitcher_games.rename(columns={'TEAM_x':'Team1'})


pitchers_team_off = pd.merge(pitcher_games, team_list, on='TEAM', how='inner')
pitchers_team_off = pitchers_team_off.rename(columns={'Team_x':'Team'})
pitchers_team_off = pd.merge(pitchers_team_off, team_list, on='Team', how='inner')
pitchers_team_off = pitchers_team_off.rename(columns={'TEAM_y':'TEAM'})

# Add opponent offense to pitching stats (Home vs Away Wrong)
#team_offense1 = team_offense.rename(columns={'TEAM': 'TEAM_OFFENSE'})
pitchers_team_offense = pd.merge(pitchers_team_off, team_offense, on='TEAM', how='inner')
#team_offense = team_offense.rename(columns={'TEAM':'Offense'})
#pitchers_team_offense = pd.merge(pitchers_team_off, team_offense, on='Offense', how='inner')

pitchers_team_offense_colu = list(pitchers_team_offense.columns)

bat_avg = bat_avg.rename(index={'Index': 'TEAM'})
bat_index = list(bat_avg.index)
bat_avg['TEAM'] = bat_index

#pitchers_team_offense = pitchers_team_offense.rename(columns={'TEAM_y':'TEAM'})
from constant_variables import TEAMS
def get_team_ini(team_name):
    for name, ini in TEAMS.items():
        if name == team_name:
            return ini

pitchers_team_mean_offense = pd.merge(pitchers_team_offense, bat_avg, on='TEAM', how='inner')
pitchers_team_mean_offense['Stadium_ini'] = list(map((lambda x:get_team_ini(x)), pitchers_team_mean_offense['Stadium']))


pitchers_team_mean_offense_colu = list(pitchers_team_mean_offense.columns)

right_cols = [s for s in pitchers_team_mean_offense_colu if "RIGHTb" in s]
left_cols = [s for s in pitchers_team_mean_offense_colu if "LEFT_b" in s]

pitchers_team_mean_left1 = pitchers_team_mean_offense.drop(right_cols, axis=1)
pitchers_team_mean_right1 = pitchers_team_mean_offense.drop(left_cols, axis=1)
            
'''
i1 = pitchers_team_mean_offense_colu.index('G_RIGHTbatter_Std')
i2 = pitchers_team_mean_offense_colu.index('G_LEFT_batter_Std')
i3 = pitchers_team_mean_offense_colu.index('SDTHB_BAT')
i4 = pitchers_team_mean_offense_colu.index('SDTHB_BATL')+1


ind = list(range(0, i1))
ind1 = list(range(i2, i4))
ind2 = ind + ind1
ind3 = list(range(0, i2))
ind4 = list(range(i3, i4))
ind5 = ind3 + ind4'''

#Seperate Batters Stats vs Hand of the Pitcher
pitchers_team_mean_left = pitchers_team_mean_left1[pitchers_team_mean_left1['throws'] == 'L']
pitchers_team_mean_right = pitchers_team_mean_right1[pitchers_team_mean_right1['throws'] == 'R']

# Incorporate Park Factors for Left and Right Handed Hitters
park_factor_left = park_factor_hand[park_factor_hand['SIDE'] == 'LHB']
park_factor_right = park_factor_hand[park_factor_hand['SIDE'] == 'RHB']
park_factor_left = park_factor_left.iloc[:, [2, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
park_factor_right = park_factor_right.iloc[:, [2, 7, 8, 9, 10, 11, 12, 13, 14, 15]]

park_factor_left = park_factor_left.rename(columns={'TEAM': 'Stadium_ini'})

pitchers_team_mean_left = pd.merge(pitchers_team_mean_left, park_factor_left, on='Stadium_ini', how='inner')
pitchers_team_mean_right = pd.merge(pitchers_team_mean_right, park_factor_left, on='Stadium_ini', how='inner')

'''
inde = list(pitchers_team_mean_left.index)
for i, team in zip(inde, pitchers_team_mean_left['TEAM']):
    pitchers_team_mean_left.loc[i, '1B_LEFT_batter_Std_Park_Adj'] = pitchers_team_mean_left.loc[i, '1B_LEFT_batter_Std']*park_factor_left['1b Factor'][park_factor_left['TEAM'] == team].item()
    pitchers_team_mean_left.loc[i, '2B_LEFT_batter_Std_Park_Adj'] = pitchers_team_mean_left.loc[i, '2B_LEFT_batter_Std']*park_factor_left['2b Factor'][park_factor_left['TEAM'] == team].item()
    pitchers_team_mean_left.loc[i, '3B_LEFT_batter_Std_Park_Adj'] = pitchers_team_mean_left.loc[i, '3B_LEFT_batter_Std']*park_factor_left['3b Factor'][park_factor_left['TEAM'] == team].item()
    pitchers_team_mean_left.loc[i, 'HR_LEFT_batter_Std_Park_Adj'] = pitchers_team_mean_left.loc[i, 'HR_LEFT_batter_Std']*park_factor_left['HR Factor'][park_factor_left['TEAM'] == team].item()
    pitchers_team_mean_left.loc[i, 'R_LEFT_batter_Std_Park_Adj'] = pitchers_team_mean_left.loc[i, 'R_LEFT_batter_Std']*park_factor_left['Runs Factor'][park_factor_left['TEAM'] == team].item()'''


pitchers_team_mean_left['1B_LEFT_batter_Std_Park_Adj'] = pitchers_team_mean_left['1B_LEFT_batter_Std']*pitchers_team_mean_left['1b Factor']
pitchers_team_mean_left['2B_LEFT_batter_Std_Park_Adj'] = pitchers_team_mean_left['2B_LEFT_batter_Std']*pitchers_team_mean_left['2b Factor']
pitchers_team_mean_left['3B_LEFT_batter_Std_Park_Adj'] = pitchers_team_mean_left['3B_LEFT_batter_Std']*pitchers_team_mean_left['3b Factor']
pitchers_team_mean_left['HR_LEFT_batter_Std_Park_Adj'] = pitchers_team_mean_left['HR_LEFT_batter_Std']*pitchers_team_mean_left['HR Factor']
pitchers_team_mean_left['R_LEFT_batter_Std_Park_Adj'] = pitchers_team_mean_left['R_LEFT_batter_Std']*pitchers_team_mean_left['Runs Factor']


'''inde1 = list(pitchers_team_mean_right.index)
for i, team in zip(inde1, pitchers_team_mean_right['TEAM']):
    pitchers_team_mean_right.loc[i, '1B_RIGHTbatter_Std_Park_Adj'] = pitchers_team_mean_right.loc[i, '1B_RIGHTbatter_Std']*park_factor_right['1b Factor'][park_factor_right['TEAM'] == team].item()
    pitchers_team_mean_right.loc[i, '2B_RIGHTbatter_Std_Park_Adj'] = pitchers_team_mean_right.loc[i, '2B_RIGHTbatter_Std']*park_factor_right['2b Factor'][park_factor_right['TEAM'] == team].item()
    pitchers_team_mean_right.loc[i, '3B_RIGHTbatter_Std_Park_Adj'] = pitchers_team_mean_right.loc[i, '3B_RIGHTbatter_Std']*park_factor_right['3b Factor'][park_factor_right['TEAM'] == team].item()
    pitchers_team_mean_right.loc[i, 'HR_RIGHTbatter_Std_Park_Adj'] = pitchers_team_mean_right.loc[i, 'HR_RIGHTbatter_Std']*park_factor_right['HR Factor'][park_factor_right['TEAM'] == team].item()
    pitchers_team_mean_right.loc[i, 'R_RIGHTbatter_Std_Park_Adj'] = pitchers_team_mean_right.loc[i, 'R_RIGHTbatter_Std']*park_factor_right['Runs Factor'][park_factor_right['TEAM'] == team].item()'''

pitchers_team_mean_right['1B_RIGHTbatter_Std_Park_Adj'] = pitchers_team_mean_right['1B_RIGHTbatter_Std']*pitchers_team_mean_right['1b Factor']
pitchers_team_mean_right['2B_RIGHTbatter_Std_Park_Adj'] = pitchers_team_mean_right['2B_RIGHTbatter_Std']*pitchers_team_mean_right['2b Factor']
pitchers_team_mean_right['3B_RIGHTbatter_Std_Park_Adj'] = pitchers_team_mean_right['3B_RIGHTbatter_Std']*pitchers_team_mean_right['3b Factor']
pitchers_team_mean_right['HR_RIGHTbatter_Std_Park_Adj'] = pitchers_team_mean_right['HR_RIGHTbatter_Std']*pitchers_team_mean_right['HR Factor']
pitchers_team_mean_right['R_RIGHTbatter_Std_Park_Adj'] = pitchers_team_mean_right['R_RIGHTbatter_Std']*pitchers_team_mean_right['Runs Factor']
'''
right_cols1 = [s for s in list(pitchers_team_mean_right.columns) if "RIGHT" in s]
left_cols1 = [s for s in list(pitchers_team_mean_left.columns) if "LEFT" in s]

left_new, left_new1 = [],[]
for hand in left_cols1:
    left_new.append(hand.replace('LEFT_', 'HAND_'))
for hand in left_new:    
    left_new1.append(hand.replace('LEFTb', 'HAND_b'))
right_new = []
for hand in right_cols1:
    right_new.append(hand.replace('RIGHT_', 'HAND_'))
right_new1 = []
for hand in right_new:    
    right_new1.append(hand.replace('RIGHTb', 'HAND_b'))

left_dict1 = dict()
for pitch, new in zip(left_cols1, left_new1):
    left_dict1[pitch] = new
    
right_dict1 = dict()
for pitch, new in zip(right_cols1, right_new1):
    right_dict1[pitch] = new'''

bat_left = ['G_LEFT_batter_Std', 'PA_LEFT_batter_Std', 'AB_LEFT_batter_Std', 'R_LEFT_batter_Std', 'H_LEFT_batter_Std', '1B_LEFT_batter_Std', '2B_LEFT_batter_Std', '3B_LEFT_batter_Std', 'HR_LEFT_batter_Std', 'TB_LEFT_batter_Std', 'SO_LEFT_batter_Std', 'BB_LEFT_batter_Std', 'HBP_LEFT_batter_Std', 'SF_LEFT_batter_Std', 'SH_LEFT_batter_Std', 'RBI_LEFT_batter_Std', 'DP_LEFT_batter_Std', 'FB%_LEFT_batter_Std', 'GB%_LEFT_batter_Std', 'LD%_LEFT_batter_Std', 'POP%_LEFT_batter_Std', 'ISO_LEFT_batter_Std', 'AVG_LEFT_batter_Std', 'OBP_LEFT_batter_Std', 'SLG_LEFT_batter_Std', 'TAV_LEFT_batter_Std', 'BASES_VS_AB_LEFT_batter_Std', '1B_LEFT_batter_Std_Park_Adj', '2B_LEFT_batter_Std_Park_Adj', '3B_LEFT_batter_Std_Park_Adj', 'HR_LEFT_batter_Std_Park_Adj', 'R_LEFT_batter_Std_Park_Adj']
left_new = []
for hand in bat_left:
    left_new.append(hand.replace('LEFT', 'HAND_ADJ'))       

bat_right = ['G_RIGHTbatter_Std', 'PA_RIGHTbatter_Std', 'AB_RIGHTbatter_Std', 'R_RIGHTbatter_Std', 'H_RIGHTbatter_Std', '1B_RIGHTbatter_Std', '2B_RIGHTbatter_Std', '3B_RIGHTbatter_Std', 'HR_RIGHTbatter_Std', 'TB_RIGHTbatter_Std', 'SO_RIGHTbatter_Std', 'BB_RIGHTbatter_Std', 'HBP_RIGHTbatter_Std', 'SF_RIGHTbatter_Std', 'SH_RIGHTbatter_Std', 'RBI_RIGHTbatter_Std', 'DP_RIGHTbatter_Std', 'FB%_RIGHTbatter_Std', 'GB%_RIGHTbatter_Std', 'LD%_RIGHTbatter_Std', 'POP%_RIGHTbatter_Std', 'ISO_RIGHTbatter_Std', 'AVG_RIGHTbatter_Std', 'OBP_RIGHTbatter_Std', 'SLG_RIGHTbatter_Std', 'TAV_RIGHTbatter_Std', 'BASES_VS_AB_RIGHTbatter_Std', '1B_RIGHTbatter_Std_Park_Adj', '2B_RIGHTbatter_Std_Park_Adj', '3B_RIGHTbatter_Std_Park_Adj', 'HR_RIGHTbatter_Std_Park_Adj', 'R_RIGHTbatter_Std_Park_Adj']      
right_new = []
for hand in bat_right:
    right_new.append(hand.replace('RIGHT', 'HAND_ADJ_'))

left_dict = dict()
for bat, new in zip(bat_left, left_new):
    left_dict[bat] = new
    
right_dict = dict()
for bat, new in zip(bat_right, right_new):
    right_dict[bat] = new
    
pitchers_team_mean_left_col = list(pitchers_team_mean_left.columns)
pitchers_team_mean_right_col = list(pitchers_team_mean_right.columns)
pitchers_team_mean_left2 = pitchers_team_mean_left.rename(columns=left_dict)
pitchers_team_mean_right2 = pitchers_team_mean_right.rename(columns=right_dict)
pitchers_team_mean_hand = pd.concat([pitchers_team_mean_left2, pitchers_team_mean_right2], axis=0, ignore_index=True)

pit_team_mean_col = list(pitchers_team_mean_right2.columns)
pit_team_mean_col1 = list(pitchers_team_mean_left2.columns)

for tr, tl in zip(pit_team_mean_col, pit_team_mean_col1):
    if tr != tl:
        print(tr, tl)
        
pitchers_team_mean_hand_col = list(pitchers_team_mean_hand.columns)

pitchers_team_mean_hand['Date_Team'] = pitchers_team_mean_hand['Date'] + ' ' + pitchers_team_mean_hand['TM']
i = 0
for i, hand in enumerate(pitchers_team_mean_hand['throws']):
    if hand == 'L':
        pitchers_team_mean_hand.loc[i, 'SDTHB_BATH'] = pitchers_team_mean_hand['SDTHB_BATL'][i]
    if hand == 'R':
        pitchers_team_mean_hand.loc[i, 'SDTHB_BATH'] = pitchers_team_mean_hand['SDTHB_BATR'][i]
        
#pitchers_team_mean_hand = pitchers_team_mean_hand.drop(['SDTHB_R', 'SDTHB_L'], axis=1)
pitchers_team_mean_hand['GB%_DEF_BAT_HAND'] = (pitchers_team_mean_hand['GB%_Def_Ef']/100) * (pitchers_team_mean_hand['GB%_HAND_ADJ_batter_Std']/100)
pitchers_team_mean_hand['FB%_DEF_BAT_HAND'] = (pitchers_team_mean_hand['FB%_Def_Ef']/100) * (pitchers_team_mean_hand['FB%_HAND_ADJ_batter_Std']/100)
pitchers_team_mean_hand['LD%_DEF_BAT_HAND'] = (pitchers_team_mean_hand['LD%_Def_Ef']/100) * (pitchers_team_mean_hand['LD%_HAND_ADJ_batter_Std']/100)
pitchers_team_mean_hand['POP%_DEF_BAT_HAND'] = (pitchers_team_mean_hand['POP%_Def_Ef']/100) * (pitchers_team_mean_hand['POP%_HAND_ADJ_batter_Std']/100)
pitchers_team_mean_hand['DP%_Def_BAT_HAND'] = (pitchers_team_mean_hand['DP%_Def_Ef']/100) * pitchers_team_mean_hand['DP_HAND_ADJ_batter_Std']


col_del_field = ['GB%_Def_Ef', 'FB%_Def_Ef', 'LD%_Def_Ef', 'POP%_Def_Ef', 'FB%_HAND_ADJ_batter_Std', 'GB%_HAND_ADJ_batter_Std', 'LD%_HAND_ADJ_batter_Std', 'POP%_HAND_ADJ_batter_Std', 'DP%_Def_BAT_HAND', 'DP_HAND_ADJ_batter_Std', 'FB Factor', 'G_Pitching_Std',
 'GB Factor', 'LD Factor', 'PU Factor', '1b Factor', '2b Factor', '3b Factor', 'HR Factor', 'Runs Factor', 'PA_batter_Std',  'G_SIDE_pit',]
pitchers_team_mean_hand = pitchers_team_mean_hand.drop(col_del_field, axis=1)

team_list = team_list.rename(columns={'TEAM': 'TEAM_OFFENSE'})
temp_team = pd.merge(wind_temp, team_list, on='TEAM_OFFENSE', how='inner')
team_list = team_list.rename(columns={'TEAM_OFFENSE': 'Oppt'})
temp_team1 = pd.merge(temp_team, team_list, on='Oppt', how='inner')

#slim_temp_team = pd.merge(slim_wind_temp, team_list, on='TEAM_OFFENSE', how='inner')


pitch_temp = pd.merge(pitchers_team_mean_hand, temp_team1, on='Date_Team', how='inner')
slim_pitch_temp = pd.merge(pitchers_team_mean_hand, slim_wind_temp, on='Date_Team', how='inner')

#slim_pitch_temp_not = player_check(pitchers_team_mean_hand,slim_wind_temp,'Date_Team','Date_Team')


col_del3 = ['Stadium_ini', 'Pitcher', 'Team', 'TEAM_x', 'Team_y', 'Date_Team', 'Game', 'Date_Opponent_num_x', 'Date_Team_num', 'Pitcher', 'Team1', 'Opponent', 'TEAM_DEFENSE_y', 'Date', 'TM', 'mlb_team_long', 'Score', 'Opponent Pitcher', 'Stadium',	'mlb_id', 'mlb_name', 'mlb_pos', 'throws', 'team_name', 'TEAM', 'IP_Pitching_Std', 'PA_Pitching_Std',	'1B_Pitching_Std',	'2B_Pitching_Std',	'3B_Pitching_Std',	'HR_Pitching_Std',	'BB_Pitching_Std',	'PA_SIDE_pit',	'1B_SIDE_pit',	'2B_SIDE_pit',	'3B_SIDE_pit',	'HR_SIDE_pit',	'BB_SIDE_pit', 'PA_Def_Ef',	'AB_Def_Ef',	'BB_Def_Ef',	'G_team_bat',	'PA_team_bat',	'AB_team_bat',	'R_team_bat',	'H_team_bat',	'2B_team_bat',	'3B_team_bat',	'HR_team_bat',	'CS_team_bat',	'BB_team_bat',	'HBP_team_bat',	'IBB_team_bat',	'Outs_team_bat',	'1B_team_bat', 'R_batter_Std', 'H_batter_Std', '1B_batter_Std',	'2B_batter_Std',	'3B_batter_Std',	'HR_batter_Std',	'BB_batter_Std',	'G_HAND_ADJ_batter_Std', 'R_HAND_ADJ_batter_Std', 'H_HAND_ADJ_batter_Std', '1B_HAND_ADJ_batter_Std', '2B_HAND_ADJ_batter_Std', '3B_HAND_ADJ_batter_Std', 'HR_HAND_ADJ_batter_Std', 'HBP_HAND_ADJ_batter_Std']
pitch_trial = pitchers_team_mean_hand.drop(col_del3, axis=1)
#pitch_trial_col = list(pitch_trial.columns)

slim_col_del3 = ['Stadium_ini', 'Pitcher', 'Team', 'TEAM_x', 'Team_y', 'Game', 'Date_Opponent_num_x', 'Date_Team_num', 'Pitcher', 'Team1', 'Opponent', 'TEAM_DEFENSE_y', 'Date', 'TM', 'mlb_team_long', 'Opponent Pitcher', 'Stadium',	'mlb_id', 'mlb_name', 'mlb_pos', 'throws', 'team_name', 'TEAM', 'IP_Pitching_Std', 'PA_Pitching_Std',	'1B_Pitching_Std',	'2B_Pitching_Std',	'3B_Pitching_Std',	'HR_Pitching_Std',	'BB_Pitching_Std',	'PA_SIDE_pit',	'1B_SIDE_pit',	'2B_SIDE_pit',	'3B_SIDE_pit',	'HR_SIDE_pit',	'BB_SIDE_pit', 'PA_Def_Ef',	'AB_Def_Ef',	'BB_Def_Ef',	'G_team_bat',	'PA_team_bat',	'AB_team_bat',	'R_team_bat',	'H_team_bat',	'2B_team_bat',	'3B_team_bat',	'HR_team_bat',	'CS_team_bat',	'BB_team_bat',	'HBP_team_bat',	'IBB_team_bat',	'Outs_team_bat',	'1B_team_bat', 'R_batter_Std', 'H_batter_Std', '1B_batter_Std',	'2B_batter_Std',	'3B_batter_Std',	'HR_batter_Std',	'BB_batter_Std',	'G_HAND_ADJ_batter_Std', 'R_HAND_ADJ_batter_Std', 'H_HAND_ADJ_batter_Std', '1B_HAND_ADJ_batter_Std', '2B_HAND_ADJ_batter_Std', '3B_HAND_ADJ_batter_Std', 'HR_HAND_ADJ_batter_Std', 'HBP_HAND_ADJ_batter_Std', 'Date_Team']
slim_pitch_trial = slim_pitch_temp.drop(slim_col_del3, axis=1)
#slim_pitch_trial_col = list(slim_pitch_trial.columns)


sdthb_park = []
for i in range(0, len(pitch_trial['R_Pitching_Std'])):
    x = (pitch_trial.loc[i, '1B_HAND_ADJ_batter_Std_Park_Adj'] + pitch_trial.loc[i, '2B_HAND_ADJ_batter_Std_Park_Adj']*int(2) + pitch_trial.loc[i, '3B_HAND_ADJ_batter_Std_Park_Adj']*int(3) + pitch_trial.loc[i, 'HR_HAND_ADJ_batter_Std_Park_Adj']*int(4) + pitch_trial.loc[i, 'BB_HAND_ADJ_batter_Std'])/pitch_trial.loc[i, 'PA_HAND_ADJ_batter_Std']
    sdthb_park.append(x)
     
slim_sdthb_park = []
for i in range(0, len(slim_pitch_trial['R_Pitching_Std'])):
    x = (slim_pitch_trial.loc[i, '1B_HAND_ADJ_batter_Std_Park_Adj'] + slim_pitch_trial.loc[i, '2B_HAND_ADJ_batter_Std_Park_Adj']*int(2) + slim_pitch_trial.loc[i, '3B_HAND_ADJ_batter_Std_Park_Adj']*int(3) + slim_pitch_trial.loc[i, 'HR_HAND_ADJ_batter_Std_Park_Adj']*int(4) + slim_pitch_trial.loc[i, 'BB_HAND_ADJ_batter_Std'])/slim_pitch_trial.loc[i, 'PA_HAND_ADJ_batter_Std']
    slim_sdthb_park.append(x)


pitch_trial['SDTHB_PARK'] = sdthb_park
slim_pitch_trial['SDTHB_PARK'] = slim_sdthb_park

col_del4 = ['1B_HAND_ADJ_batter_Std_Park_Adj', '2B_HAND_ADJ_batter_Std_Park_Adj', '3B_HAND_ADJ_batter_Std_Park_Adj', 'HR_HAND_ADJ_batter_Std_Park_Adj', 'PA_HAND_ADJ_batter_Std', 'BB_HAND_ADJ_batter_Std']
pitch_trial = pitch_trial.drop(col_del4, axis=1)
#pitch_trial_col1 = list(pitch_trial.columns)

col_del4 = ['1B_HAND_ADJ_batter_Std_Park_Adj', '2B_HAND_ADJ_batter_Std_Park_Adj', '3B_HAND_ADJ_batter_Std_Park_Adj', 'HR_HAND_ADJ_batter_Std_Park_Adj', 'PA_HAND_ADJ_batter_Std', 'BB_HAND_ADJ_batter_Std']
slim_pitch_trial = slim_pitch_trial.drop(col_del4, axis=1)
#slim_pitch_trial_col1 = list(slim_pitch_trial.columns)

pitch_trial.info()
slim_pitch_trial.info()


y = pd.Series(pitchers_team_mean_hand['Score'], dtype='int', name='Score')
y1 = pd.Series(slim_pitch_trial['Score'], dtype='int', name='Score')
slim_pitch_trial = slim_pitch_trial.drop('Score', axis=1)

pitch_trial = pitch_trial.drop('HAND_pit', axis=1)
slim_pitch_trial = slim_pitch_trial.drop('HAND_pit', axis=1)


for row in pitch_trial:
    pitch_trial[row] = list(map((lambda x:round(float(x),2)), pitch_trial[row]))



'''col_del5 = ['Date_Opponent_num_x', 'Date', 'Team', 'mlb_id', 'mlb_name', 'mlb_pos', 'TEAM_DEFENSE_y', 'mlb_team_long', 'throws', 'TEAM_x', 'TEAM_x', 'Team_y', 'Date_Team_num']
pitchers_team_mean_hand_clean = pitchers_team_mean_hand.drop(col_del5, axis=1)
pitchers_team_mean_hand_clean_col = pitchers_team_mean_hand_clean.columns.tolist()
pitchers_team_mean_hand_clean_col = pitchers_team_mean_hand_clean_col[-1:] + pitchers_team_mean_hand_clean_col[:-1]
pitchers_team_mean_hand_clean = pitchers_team_mean_hand_clean[pitchers_team_mean_hand_clean_col]

pitchers_team_mean_hand_clean_col.insert(1, pitchers_team_mean_hand_clean_col.pop(4))
pitchers_team_mean_hand_clean_col.insert(6, pitchers_team_mean_hand_clean_col.pop(86))
pitchers_team_mean_hand_clean = pitchers_team_mean_hand_clean[pitchers_team_mean_hand_clean_col]'''

pitch_trial.to_csv('C:/Users/mrcrb/source/repos/MLB/pitch_trial.csv', sep=',', encoding='utf-8', index=False)
slim_pitch_trial.to_csv('C:/Users/mrcrb/source/repos/MLB/slim_pitch_trial.csv', sep=',', encoding='utf-8', index=False)

y.to_csv('C:/Users/mrcrb/source/repos/MLB/scores.csv', sep=',', encoding='utf-8', index=False)
y1.to_csv('C:/Users/mrcrb/source/repos/MLB/scores1.csv', sep=',', encoding='utf-8', index=False)

pitch_trial.to_csv("C:/Users/mrcrb/PythonScripts/MLB/MLB/pitch_trial.csv", sep=',', encoding='utf-8', index=False)
y.to_csv("C:/Users/mrcrb/PythonScripts/MLB/MLB/scores.csv", sep=',', encoding='utf-8', index=False)

'''
print("Scrape Complete")
f.write('Finishing Script')
f.close()
'''