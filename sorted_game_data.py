import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from mlb_classes1 import Players
from open_dict import daily_lineups_pitch, date

# Need to fix information on traded players (Team, Park, Defensive Data)

df = pd.read_csv("MLB/game_date_info1.csv", encoding='latin-1')
df['date'] = list(map((lambda x:date(x)), df['date']))
df['Date_Team'] = list(map((lambda x,y:x + ' ' + y), df['date'], df['team_home']))
df['Date_Team_H'] = list(map((lambda x,y:x + ' ' + y), df['date'], df['team_away']))


from constant_variables import g_df_away_titles, g_df_away_cols
game_dataframe_away = Players(data = df, column_names = g_df_away_cols, column_titles = g_df_away_titles)

game_dataframe_home = Players(data = df, 
                       column_names = ['Date_Team_H', 'date', 'team_home', 'team_away', 'pitcher_away', 'expected_runs_home', 'actual_runs_home', 
                                       'temp', 'humidity', 'rain', 'league'],
                       column_titles = ['Date_Team', 'Date', 'Team_Bat', 'Stadium', 'Team_Pitch', 'Expected_Runs', 'Runs', 'Temp', 
                                        'Humidity', 'Rain', 'League'])


# Create Instance from Class
game_frames_away = game_dataframe_away.create_df()
game_frames_home = game_dataframe_home.create_df()
game_frames = pd.concat([game_frames_away,game_frames_home])

game_dataframe = Players(data = game_frames, 
                       column_names = ['Date_Team', 'Date', 'Team_Bat', 'Stadium', 'Team_Pitch', 'Expected_Runs', 'Runs', 'Temp', 
                                        'Humidity', 'Rain', 'League'],
                       column_titles = ['Date_Team', 'Date', 'Team_Bat', 'Stadium', 'Team_Pitch', 'Expected_Runs', 'Runs', 'Temp', 
                                        'Humidity', 'Rain', 'League'])

# Create Dictionary from Data
game_dic1 = game_dataframe.create_dict('Date_Team')
game_dict_list = game_dic1[1]
game_dict_dates = game_dic1[2]


df1 = pd.read_csv("MLB/batter_all.csv", encoding='latin-1')
df1['mlb_team_long'] = list(map((lambda x:x.upper()), df1['mlb_team_long']))
div_9 = ['R_batter_Std', 'SO_batter_Std', 'RBI_batter_Std', 'DP_batter_Std', 'R_RIGHTbatter_Std', 
        'TB_RIGHTbatter_Std', 'SO_RIGHTbatter_Std', 'RBI_RIGHTbatter_Std', 'DP_RIGHTbatter_Std', 'R_LEFT_batter_Std', 
        'TB_LEFT_batter_Std', 'SO_LEFT_batter_Std', 'RBI_LEFT_batter_Std', 'DP_LEFT_batter_Std']
for div in div_9:
    df1[div] = round(df1[div]/df1['PA_batter_Std'],2)

player_frame = Players(data = df1, 
                       column_names = ['mlb_name', 'mlb_id', 'mlb_pos', 'mlb_team_long', 'TEAM', 'bats', 'LG_batter_Std',
                                                  'PA_batter_Std', 'R_batter_Std', 'SO_batter_Std', 'RBI_batter_Std', 'DP_batter_Std', 'AVG_batter_Std', 'OBP_batter_Std', 'SLG_batter_Std', 'OPS_batter_Std', 'ISO_batter_Std',
                                                  'oppOPS_batter_Std', 'TAv_batter_Std', 'VORP_batter_Std', 'FRAA_batter_Std', 
                                                  'BWARP_batter_Std', 'BASES_VS_AB_batter_Std', 'SDTHB_BAT'],
                       column_titles = ['player_name', 'player_id', 'player_position', 'team', 'team_ini', 'bat_hand', 'league',
                                        'batter_PA', 'batter_runs', 'batter_SO', 'batter_RBI', 'batter_DP', 'batter_AVG', 'batter_OBP', 
                                        'batter_SLG', 'batter_OPS', 'batter_ISO', 'batter_oppOPS', 'batter_TAv', 'batter_VORP', 
                                        'batter_FRAA', 'batter_BWARP', 'batter_BASES_VS_AB', 'hit_mult'])

player_frame_right = Players(data = df1, 
                             column_names = ['mlb_name', 'mlb_pos', 'mlb_team_long', 'TEAM', 'bats', 'LG_batter_Std',
                                             'G_RIGHTbatter_Std', 'PA_RIGHTbatter_Std', 'R_RIGHTbatter_Std', 'TB_RIGHTbatter_Std', 
                                             'SO_RIGHTbatter_Std', 'RBI_RIGHTbatter_Std', 'DP_RIGHTbatter_Std', 'FB%_RIGHTbatter_Std', 
                                             'GB%_RIGHTbatter_Std', 'LD%_RIGHTbatter_Std', 'POP%_RIGHTbatter_Std', 'ISO_RIGHTbatter_Std', 
                                             'AVG_RIGHTbatter_Std', 'OBP_RIGHTbatter_Std', 'SLG_RIGHTbatter_Std', 'TAV_RIGHTbatter_Std', 
                                             'BASES_VS_AB_RIGHTbatter_Std', 'SDTHB_BATR'],
                             column_titles = ['player_name', 'player_position', 'team', 'team_ini', 'bat_hand', 'league',
                                             'bat_games_R', 'bat_PA_R', 'bat_runs_R', 'bat_TB_R', 'bat_SO_R', 'bat_RBI_R', 
                                             'bat_DP_R', 'bat_FB%_R', 'bat_GB%_R', 'bat_LD%_R', 'bat_POP%_R', 'bat_ISO_R', 
                                             'bat_AVG_R', 'bat_OBP_R', 'bat_SLG_R', 'bat_TAV_R', 'batter_BASES_VS_AB_R', 
                                             'hit_mult_R'])
                
player_frame_left = Players(data = df1, 
                             column_names = ['mlb_name', 'mlb_pos', 'mlb_team_long', 'TEAM', 'bats', 'LG_batter_Std',
                                             'G_LEFT_batter_Std', 'PA_LEFT_batter_Std', 'R_LEFT_batter_Std', 'TB_LEFT_batter_Std', 
                                             'SO_LEFT_batter_Std', 'RBI_LEFT_batter_Std', 'DP_LEFT_batter_Std', 'FB%_LEFT_batter_Std', 
                                             'GB%_LEFT_batter_Std', 'LD%_LEFT_batter_Std', 'POP%_LEFT_batter_Std', 'ISO_LEFT_batter_Std', 
                                             'AVG_LEFT_batter_Std', 'OBP_LEFT_batter_Std', 'SLG_LEFT_batter_Std', 'TAV_LEFT_batter_Std', 
                                             'BASES_VS_AB_LEFT_batter_Std', 'SDTHB_BATR'],
                             column_titles = ['player_name', 'player_position', 'team', 'team_ini', 'bat_hand', 'league',
                                             'bat_games_L', 'bat_PA_L', 'bat_runs_L', 'bat_TB_L', 'bat_SO_L', 'bat_RBI_L', 
                                             'bat_DP_L', 'bat_FB%_L', 'bat_GB%_L', 'bat_LD%_L', 'bat_POP%_L', 'bat_ISO_L', 
                                             'bat_AVG_L', 'bat_OBP_L', 'bat_SLG_L', 'bat_TAV_L', 'batter_BASES_VS_AB_L', 
                                             'hit_mult_L'])


# Create Instance from Class
player_frames = player_frame.create_df()
player_frames_right = player_frame_right.create_df()
player_frames_left =player_frame_left.create_df()

# Create Dictionary from Data
player_dic = player_frame.create_dict('mlb_name')
player_dict_tot = player_dic[1]
player_dict_stats = player_dic[2]

# Create Dictionary of batters vs righty pitchers
player_dic_right = player_frame_right.create_dict('mlb_name')
player_dict_tot_right = player_dic_right[1]
player_dict_stats_right = player_dic_right[2]

# Create Dictionary of batters vs lefty pitchers
player_dic_left = player_frame_left.create_dict('mlb_name')
player_dict_tot_left = player_dic_left[1]
player_dict_stats_left = player_dic_left[2]

df2 = pd.read_csv("MLB/pitcher_all.csv", encoding='latin-1')
df2['SO_Pitching_Std'] = round(df2['SO_Pitching_Std']/df2['PA_Pitching_Std'],2)
df2['G_Pitching_Std'] = round(df2['IP_Pitching_Std']/df2['G_Pitching_Std'],2)
df2['G_Pitching_Std'] = 1 - df2['G_Pitching_Std']/9


pitcher_frame = Players(data = df2, 
                       column_names = ['mlb_name', 'mlb_id', 'mlb_pos', 'mlb_team_long', 'TEAM', 'throws', 'IP_Pitching_Std',
                                       'PA_Pitching_Std', 'SO_Pitching_Std', 'DRA_Pitching_Std', 'ERA_Pitching_Std',
                                       'PPF_Pitching_Std', 'VORP_Pitching_Std', 'FIP_Pitching_Std', 'PVORP_Pitching_Std',
                                       'PWARP_Pitching_Std', 'BASES_VS_AB_Pitching_Std', 'SDTHB_ALL', 'H_Def_Ef', 'HR_Def_Ef', 
                                       'GB%_Def_Ef', 'FB%_Def_Ef', 'LD%_Def_Ef', 'POP%_Def_Ef', 'DP%_Def_Ef'],
                       column_titles = ['player_name', 'player_id', 'player_position', 'team', 'team_ini', 'pitcher_hand', 'pitcher_IP',
                                        'pitcher_PA', 'pitcher_SO', 'pitcher_DRA', 'pitcher_ERA', 'pitcher_PPF', 'pitcher_VORP', 
                                        'pitcher_FIP', 'pitcher_PVORP', 'pitcher_PWARP', 'pitcher_BASES_VS_AB', 'pitcher_hit_mult',
                                        'def_H', 'def_HR', 'def_GB%', 'def_FB%', 'def_LD%', 'def_POP%', 'def_DP%'])


# Create Instance from Class
pitcher_frames = pitcher_frame.create_df()

# Create Dictionary from Data
pitcher_dic = pitcher_frame.create_dict('mlb_name')
pitcher_dict_tot = pitcher_dic[1]
pitcher_dict_stats = pitcher_dic[2]


column_names_left = ['mlb_name', 'mlb_pos', 'mlb_team_long', 'TEAM', 'bats', 'LG_batter_Std',
                                             'G_LEFT_batter_Std', 'PA_LEFT_batter_Std', 'R_LEFT_batter_Std', 'TB_LEFT_batter_Std', 
                                             'SO_LEFT_batter_Std', 'RBI_LEFT_batter_Std', 'DP_LEFT_batter_Std', 'FB%_LEFT_batter_Std', 
                                             'GB%_LEFT_batter_Std', 'LD%_LEFT_batter_Std', 'POP%_LEFT_batter_Std', 'ISO_LEFT_batter_Std', 
                                             'AVG_LEFT_batter_Std', 'OBP_LEFT_batter_Std', 'SLG_LEFT_batter_Std', 'TAV_LEFT_batter_Std', 
                                             'BASES_VS_AB_LEFT_batter_Std', 'SDTHB_BATR']

column_names_right = ['mlb_name', 'mlb_pos', 'mlb_team_long', 'TEAM', 'bats', 'LG_batter_Std',
                                             'G_RIGHTbatter_Std', 'PA_RIGHTbatter_Std', 'R_RIGHTbatter_Std', 'TB_RIGHTbatter_Std', 
                                             'SO_RIGHTbatter_Std', 'RBI_RIGHTbatter_Std', 'DP_RIGHTbatter_Std', 'FB%_RIGHTbatter_Std', 
                                             'GB%_RIGHTbatter_Std', 'LD%_RIGHTbatter_Std', 'POP%_RIGHTbatter_Std', 'ISO_RIGHTbatter_Std', 
                                             'AVG_RIGHTbatter_Std', 'OBP_RIGHTbatter_Std', 'SLG_RIGHTbatter_Std', 'TAV_RIGHTbatter_Std', 
                                             'BASES_VS_AB_RIGHTbatter_Std', 'SDTHB_BATR']

column_names_all = ['mlb_name', 'mlb_id', 'mlb_pos', 'mlb_team_long', 'TEAM', 'bats', 'LG_batter_Std',
                                                  'PA_batter_Std', 'R_batter_Std', 'SO_batter_Std', 'RBI_batter_Std', 'DP_batter_Std', 'AVG_batter_Std', 'OBP_batter_Std', 'SLG_batter_Std', 'OPS_batter_Std', 'ISO_batter_Std',
                                                  'oppOPS_batter_Std', 'TAv_batter_Std', 'VORP_batter_Std', 'FRAA_batter_Std', 
                                                  'BWARP_batter_Std', 'BASES_VS_AB_batter_Std', 'SDTHB_BAT']

column_names_new = ['mlb_name', 'mlb_pos', 'mlb_team_long', 'TEAM', 'bats', 'LG_batter', 'G_batter',
                                                  'PA_batter', 'R_batter', 'TB_batter', 'SO_batter', 'RBI_batter', 'DP_batter', 'FB%_batter', 'GB%_batter', 'LD%_batter', 'POP%_batter', 'ISO_batter',
                                                  'AVG_batter', 'OBP_batter', 'SLG_batter', 'TAV_batter', 
                                                  'BASES_VS_AB_batter', 'SDTHB_BAT']

df1_left_index = [list(df1.columns).index(name) for name in column_names_left]  
df1_right_index = [list(df1.columns).index(name) for name in column_names_right]
df1_all_index = [list(df1.columns).index(name) for name in column_names_all]

'''
df1_left_index = []
for name in column_names_left:
    ind = list(df1.columns).index(name)
    df1_left_index.append(ind)
    
df1_right_index = []
for name in column_names_right:
    ind = list(df1.columns).index(name)
    df1_right_index.append(ind)
    

df1_all_index = []
for name in column_names_all:
    ind = list(df1.columns).index(name)
    df1_all_index.append(ind)
'''

# Create Lineups
player_lineups = player_frame.create_lineups(df1_all_index, df1_right_index, df1_left_index)
player_lineup = player_lineups[0]
player_lineup_right = player_lineups[1]
player_lineup_left = player_lineups[2]
player_lineup_right_alt = player_lineups[3]
player_lineup_left_alt = player_lineups[4]


#for k,v in player_lineup.items():
    #v.columns = column_names_new
for k,v in player_lineup_right.items():
    v.columns = column_names_new
for k,v in player_lineup_left.items():
    v.columns = column_names_new
for k,v in player_lineup_right_alt.items():
    v.columns = column_names_new
for k,v in player_lineup_left_alt.items():
    v.columns = column_names_new

# Based on if team is in AL or NL(not on stadium) - Interleague games won't be accurate
def team_mean_batting(player_dic):
    team_mean = {}
    for team in list(player_dic.keys()):
        team_info = {k:v for k,v in zip(list(player_dic[team].keys()[6:]), list(player_dic[team].iloc[:,6:].mean()))} 
        team_mean[team] = team_info
        team_info = {}
    return team_mean


'''
# Based on if team is in AL or NL(not on stadium) - Interleague games won't be accurate
def team_mean_batting(player_dic):
    team_mean, team_info = {},{}
    for team in list(player_dic.keys()):
        for k,v in zip(list(player_dic[team].keys()[6:]), list(player_dic[team].iloc[:,6:].mean())):
            team_info[k] = v
        team_mean[team] = team_info
        team_info = {}
    return team_mean
'''
    
team_mean = team_mean_batting(player_lineup)


'''game_pitcher_stats_dict = {}
for k,v in game_dict_list.items():
    for k1,v1 in v.items():
        pitcher = game_dict_list[k]['Opponent_Pitcher']
        opponent = game_dict_list[k]['Opponent']
        if pitcher in list(pitcher_dict_stats.keys()) and opponent == pitcher_dict_stats[pitcher]['team'].upper():
                game_pitcher_stats_dict[k] = {**game_dict_list[k], **pitcher_dict_stats[pitcher]}'''

from constant_variables import LEAGUE
def get_team_league(team_ini):
    for ini, league in LEAGUE.items():
        if ini == team_ini:
            return league

game_pitcher_stats_dict = {}
for k,v in game_dict_dates.items():
    for k1,v1 in v.items():
        pitcher = game_dict_dates[k]['Team_Pitch']
        opponent = game_dict_dates[k]['Stadium']
        if pitcher in list(pitcher_dict_stats.keys()) and opponent == pitcher_dict_stats[pitcher]['team_ini'].upper():
            game_pitcher_stats_dict[k] = {**game_dict_dates[k], **pitcher_dict_stats[pitcher]}


game_pitcher_batter_stats_dict, team_means = {},{}
team_mean_left = team_mean_batting(player_lineup_left)
team_mean_right = team_mean_batting(player_lineup_right)
team_mean_left_alt = team_mean_batting(player_lineup_left_alt)
team_mean_right_alt = team_mean_batting(player_lineup_right_alt)
for k,v in game_pitcher_stats_dict.items():
    for k1,v1 in v.items():
        team_ini = game_pitcher_stats_dict[k]['Team_Bat']
        if game_pitcher_stats_dict[k]['pitcher_hand'] == 'L' and get_team_league(game_pitcher_stats_dict[k]['Team_Bat']) == 'AL' and get_team_league(game_pitcher_stats_dict[k]['Stadium']) == 'AL':
            team_means[team_ini] = team_mean_left[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}
        elif game_pitcher_stats_dict[k]['pitcher_hand'] == 'R' and get_team_league(game_pitcher_stats_dict[k]['Team_Bat']) == 'AL' and get_team_league(game_pitcher_stats_dict[k]['Stadium']) == 'AL':
            team_means[team_ini] = team_mean_right[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}
        elif game_pitcher_stats_dict[k]['pitcher_hand'] == 'L' and get_team_league(game_pitcher_stats_dict[k]['Team_Bat']) == 'NL' and get_team_league(game_pitcher_stats_dict[k]['Stadium']) == 'AL':
            team_means[team_ini] = team_mean_left_alt[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}
        elif game_pitcher_stats_dict[k]['pitcher_hand'] == 'R' and get_team_league(game_pitcher_stats_dict[k]['Team_Bat']) == 'AL' and get_team_league(game_pitcher_stats_dict[k]['Stadium']) == 'NL':
            team_means[team_ini] = team_mean_right_alt[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}
        elif game_pitcher_stats_dict[k]['pitcher_hand'] == 'L' and get_team_league(game_pitcher_stats_dict[k]['Team_Bat']) == 'NL' and get_team_league(game_pitcher_stats_dict[k]['Stadium']) == 'NL':
            team_means[team_ini] = team_mean_left[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}
        elif game_pitcher_stats_dict[k]['pitcher_hand'] == 'R' and get_team_league(game_pitcher_stats_dict[k]['Team_Bat']) == 'NL' and get_team_league(game_pitcher_stats_dict[k]['Stadium']) == 'NL':
            team_means[team_ini] = team_mean_right[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}
        elif game_pitcher_stats_dict[k]['pitcher_hand'] == 'L' and get_team_league(game_pitcher_stats_dict[k]['Team_Bat']) == 'AL' and get_team_league(game_pitcher_stats_dict[k]['Stadium']) == 'NL':
            team_means[team_ini] = team_mean_left_alt[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}
        elif game_pitcher_stats_dict[k]['pitcher_hand'] == 'R' and get_team_league(game_pitcher_stats_dict[k]['Team_Bat']) == 'NL' and get_team_league(game_pitcher_stats_dict[k]['Stadium']) == 'AL':
            team_means[team_ini] = team_mean_right_alt[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}


game_logs = pd.DataFrame(list(game_pitcher_batter_stats_dict.values()), columns = list(list(game_pitcher_batter_stats_dict.values())[0].keys()))
game_logs['pitcher_BASES_VS_AB'] = list(map((lambda x:round(float(x),2)), game_logs['pitcher_BASES_VS_AB']))
game_logs['pitcher_hit_mult'] = list(map((lambda x:round(float(x),2)), game_logs['pitcher_hit_mult']))
game_logs['Expected_Runs'][287] = float(3.0)
game_logs['Expected_Runs'] = game_logs['Expected_Runs'].astype(float)
game_logs = game_logs.groupby(game_logs.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))


ind = []
for i in range(16,33):
    ind.append(i)
ind1 = []    
for i in range(35,51):
    ind1.append(i)
ind2 = [4,6,7,8]+ind+ind1

for i in ind2:
    game_logs.iloc[:, i] = game_logs.iloc[:, i].astype(float)

y = game_logs['Runs'].astype(int)
X = game_logs.iloc[:,ind2]


from constant_variables import PLAYERS
def get_team_player(player_name):
    for player, name in PLAYERS.items():
        if player == player_name:
            #player_name = name
            return name
            break
    return player_name
        
        
        
daily_lineups = daily_lineups_pitch()
unknown, logs = [], []
for d in game_logs['Date'].unique():
    for k,v in daily_lineups[d].items():
            for k1,v1 in v.items():
                if len(list(game_logs['pitcher_hand'][game_logs['Team_Bat']==k][game_logs['Date']==d]))>0 and list(game_logs['pitcher_hand'][game_logs['Team_Bat']==k][game_logs['Date']==d])[0] == 'R':
                    try:
                        daily_lineups[d][k][k1] = player_dict_stats_right[k1.upper()]
                    except KeyError:
                        try:
                            logs.append(['RIGHT0:' + d,k,k1])
                            k2 = get_team_player(k1)
                            #daily_lineups[d][k][k2] = daily_lineups[d][k].pop(k1, None)
                            daily_lineups[d][k][k1] = player_dict_stats_right[k2.upper()]
                            logs.append(['RIGHT:' + d,k,k1,k2])
                        except KeyError:
                            unknown.append(k1)
                            logs.append(['RIGHT1:' + d,k,k1,k2])
                else:
                    try:
                        daily_lineups[d][k][k1] = player_dict_stats_left[k1.upper()]
                    except KeyError:
                        try:
                            logs.append(['LEFT0:' + d,k,k1])
                            k2 = get_team_player(k1)
                            #daily_lineups[d][k][k2] = daily_lineups[d][k].pop(k1, None)
                            daily_lineups[d][k][k1] = player_dict_stats_left[k2.upper()]
                            logs.append(['LEFT:' + d,k,k1,k2])
                        except KeyError:
                            unknown.append(k1)
                            logs.append(['LEFT1:' + d,k,k1,k2])
unknown_uniq = []
for un in unknown:
    if un not in unknown_uniq:
        unknown_uniq.append(un)

unknown_info = []
for k,v in daily_lineups.items():
    for k1,v1 in v.items():
        for k2,v2 in v1.items():
            if k2 in unknown_uniq:
                unknown_info.append([k,k1,k2])

for un in unknown_info:                
    daily_lineups[un[0]][un[1]].pop(un[2], None)
                

daily_lineups_mean, daily_lineups_mean1, day, list_team, list_team_mean, list_mean, dates, list_teams = {}, {}, {}, [], [], [], [], []
for k,v in daily_lineups.items():
    for k1,v1 in v.items():
        for k2,v2 in v1.items():
            list_team.append(list(v2.values())[9:])
        hitters_game = pd.DataFrame(np.column_stack([list_team]), columns = list(v2.keys())[9:], dtype = float)
        daily_lineups_mean[k + ' ' + k1] = hitters_game.mean()
        for lk, lv in zip(list(v2.keys())[9:], list(hitters_game.mean())):
                day[lk] = lv
        daily_lineups_mean1[k + ' ' + k1] = day
        list_team, list_mean, list_teams, day = [], [], [], {}
        
        
game_pit_bat_lineup_dic = {}        
for k,v in game_pitcher_batter_stats_dict.items():
    if k in daily_lineups_mean1.keys():
        game_pit_bat_lineup_dic[k] = {**game_pitcher_batter_stats_dict[k], **daily_lineups_mean1[k]}

col_game =  ['bat_SO_R',
 'bat_RBI_R',
 'bat_DP_R',
 'bat_FB%_R',
 'bat_GB%_R',
 'bat_LD%_R',
 'bat_POP%_R',
 'bat_ISO_R',
 'bat_AVG_R',
 'bat_OBP_R',
 'bat_SLG_R',
 'bat_TAV_R',
 'batter_BASES_VS_AB_R',
 'hit_mult_R',
 'bat_SO_L',
 'bat_RBI_L',
 'bat_DP_L',
 'bat_FB%_L',
 'bat_GB%_L',
 'bat_LD%_L',
 'bat_POP%_L',
 'bat_ISO_L',
 'bat_AVG_L',
 'bat_OBP_L',
 'bat_SLG_L',
 'bat_TAV_L',
 'batter_BASES_VS_AB_L',
 'hit_mult_L']

col_new =  ['bat_SO',
 'bat_RBI',
 'bat_DP',
 'bat_FB%',
 'bat_GB%',
 'bat_LD%',
 'bat_POP%',
 'bat_ISO',
 'bat_AVG',
 'bat_OBP',
 'bat_SLG',
 'bat_TAV',
 'batter_BASES_VS_AB',
 'hit_mult',
 'bat_SO',
 'bat_RBI',
 'bat_DP',
 'bat_FB%',
 'bat_GB%',
 'bat_LD%',
 'bat_POP%',
 'bat_ISO',
 'bat_AVG',
 'bat_OBP',
 'bat_SLG',
 'bat_TAV',
 'batter_BASES_VS_AB',
 'hit_mult']

col_new1 =  ['bat_SO',
 'bat_RBI',
 'bat_DP',
 'bat_FB%',
 'bat_GB%',
 'bat_LD%',
 'bat_POP%',
 'bat_ISO',
 'bat_AVG',
 'bat_OBP',
 'bat_SLG',
 'bat_TAV',
 'batter_BASES_VS_AB',
 'hit_mult']

game_pit_bat_lineup_dic2, new_dict = {}, {}
for k,v in game_pit_bat_lineup_dic.items():
    for k1,v1 in v.items():
        if k1 in col_game:
            ind = col_game.index(k1)
            new_dict[col_new[ind]] = game_pit_bat_lineup_dic[k][k1]
        else:
            new_dict[k1] = game_pit_bat_lineup_dic[k][k1]
    game_pit_bat_lineup_dic2[k] = new_dict
    new_dict = {}
        
game_pit_bat_lineup_dic1 = game_pit_bat_lineup_dic

game_logs1 = pd.DataFrame(list(game_pit_bat_lineup_dic2.values()), columns = list(list(game_pit_bat_lineup_dic2.values())[0].keys()))
'''game_logs['pitcher_BASES_VS_AB'] = list(map((lambda x:round(float(x),2)), game_logs['pitcher_BASES_VS_AB']))
game_logs['pitcher_hit_mult'] = list(map((lambda x:round(float(x),2)), game_logs['pitcher_hit_mult']))
game_logs['Expected_Runs'][287] = float(3.0)
game_logs['Expected_Runs'] = game_logs['Expected_Runs'].astype(float)'''

def game_nan_index(column):
    indicies = []
    col_list = list(column.values)
    for i,game in enumerate(col_list): 
        if game == 'nan':
            indicies.append(i)
    return indicies
            
inde1 = game_nan_index(game_logs1['Expected_Runs'])
game_logs1['Expected_Runs'][inde1] = float(3.5)
##game_logs1 = game_logs1.groupby(game_logs1.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))


ind = []
for i in range(17,33):
    ind.append(i)
ind1 = []    
for i in range(36,66):
    ind1.append(i)
ind2 = [4,6,7,8]+ind+ind1

for i in ind2:
    game_logs1.iloc[:, i] = game_logs1.iloc[:, i].astype(float)
    game_logs1.iloc[:, i] = round(game_logs1.iloc[:, i],2)

y = game_logs1['Runs'].astype(int)
X = game_logs1.iloc[:,ind2]
        

df4 = pd.read_csv("MLB/pitcher_vs_side.csv", encoding='latin-1')
df4 = df4.iloc[:,1:]

from constant_variables import pit_side_col_list
pitcher_side_frame = Players(data = df4, 
                       column_names = pit_side_col_list,
                       column_titles = pit_side_col_list)


# Create Instance from Class
pitcher_side_frames = pitcher_side_frame.create_df()

# Create Dictionary from Data
pitcher_side_dic = pitcher_side_frame.create_dict('NAME')
pitcher_side_dict_tot = pitcher_side_dic[1]
pitcher_side_dict_stats = pitcher_side_dic[2]

test = []
daily_lineups_mean_pit, daily_lineups_mean1_pit, day_pit, list_team_pit, list_team_mean_pit, list_mean_pit, dates_pit, list_teams_pit = {}, {}, {}, [], [], [], [], []
for k,v in daily_lineups.items():
    for k1,v1 in v.items():
        for k2,v2 in v1.items():
            try:    
                team_pit = game_pitcher_batter_stats_dict[k + ' ' + k1]['Team_Pitch']
                hand_pit = game_pitcher_batter_stats_dict[k + ' ' + k1]['pitcher_hand']
            except KeyError:
                break
            bat_hand = list(v2.values())[3]
            if bat_hand == 'S' and hand_pit == 'L':
                bat_hand = 'R'
            elif bat_hand == 'S' and hand_pit == 'R':
                bat_hand = 'L'
            if team_pit.upper() in list(pitcher_side_frames['NAME']) and pitcher_side_frames.iloc[:,5:][pitcher_side_frames['NAME']==team_pit][pitcher_side_frames['HAND_pit']==bat_hand].empty == False:
                list_team_pit.append(pitcher_side_frames.iloc[:,5:][pitcher_side_frames['NAME']==team_pit][pitcher_side_frames['HAND_pit']==bat_hand])
                dates_pit.append(team_pit)
            else:
                test.append([team_pit, bat_hand, k2, k1, k, bat_hand])
        if len(list_team_pit) > 0:
            pitchers_game = pd.DataFrame(pd.concat([li for li in list_team_pit]), columns = list(pitcher_side_frames.iloc[:,5:].columns), dtype = float)
            daily_lineups_mean_pit[k + ' ' + k1] = pitchers_game.mean()
            # moved forward 1 tab
            for lk, lv in zip(list(pitcher_side_frames.iloc[:,5:].columns), list(pitchers_game.mean())):
                day_pit[lk] = lv
            daily_lineups_mean1_pit[k + ' ' + k1] = day_pit
        list_team_pit, list_mean_pit, list_teams_pit, day_pit = [], [], [], {}
        
game_pitcher_bat_lineup_dic = {}        
for k,v in game_pit_bat_lineup_dic2.items():
    if k in daily_lineups_mean1_pit.keys():
        game_pitcher_bat_lineup_dic[k] = {**game_pit_bat_lineup_dic2[k], **daily_lineups_mean1_pit[k]}

df4 = pd.read_csv("MLB/bullpen.csv", encoding='latin-1')
bull = {}
game_pitcher_bat_lineup_dic1 = game_pitcher_bat_lineup_dic
#game_pitcher_bat_lineup_dic = game_pitcher_bat_lineup_dic1
game_pitcher_bat_lineup_bullpen_dic = {}        
for k,v in game_pitcher_bat_lineup_dic.items():
    pitcher = v['Team_Pitch']
    team = v['team_ini']
    for col in list(df4.columns[1:-1]):
        ratio = df2['G_Pitching_Std'][df2['mlb_name'] == pitcher].values[0]
        #bull[col] = df4[col][df4['Team_Ini']==team].values[0]
        game_pitcher_bat_lineup_dic[k][col] = df4[col][df4['Team_Ini']==team].values[0]*ratio

game_logs2 = pd.DataFrame(list(game_pitcher_bat_lineup_dic.values()), columns = list(list(game_pitcher_bat_lineup_dic.values())[0].keys()))

def game_nan_index(column):
    indicies = []
    col_list = list(column.values)
    for i,game in enumerate(col_list): 
        if game == 'nan':
            indicies.append(i)
    return indicies
            
inde2 = game_nan_index(game_logs2['Expected_Runs'])
game_logs2['Expected_Runs'][inde1] = float(3.5)


ind = []
for i in range(17,34):
    ind.append(i)
ind1 = []    
for i in range(36,110):
    ind1.append(i)
ind2 = [4,6,7,8]+ind+ind1

for i in ind2:
    game_logs2.iloc[:, i] = game_logs2.iloc[:, i].astype(float)
    game_logs2.iloc[:, i] = round(game_logs2.iloc[:, i],2)

y = game_logs2['Runs'].astype(int)
X = game_logs2.iloc[:,ind2]
X1 = game_logs2.iloc[:,[1]+ind2]

X.to_csv("C:/Users/mrcrb/PythonScripts/MLB/MLB/X_sort.csv", sep=',', encoding='utf-8', index=False)
y.to_csv("C:/Users/mrcrb/PythonScripts/MLB/MLB/y_sort.csv", sep=',', encoding='utf-8', index=False)
X1.to_csv("C:/Users/mrcrb/PythonScripts/MLB/MLB/X1_sort.csv", sep=',', encoding='utf-8', index=False)


# Compiling future game data that will be predicted
df5 = pd.read_csv("MLB/future_games.csv", encoding='latin-1')
ones = pd.DataFrame(np.ones((len(df5),12)), columns = ['pitcher_IP', 'pitcher_PA', 'pitcher_SO', 'pitcher_DRA', 'pitcher_ERA', 'pitcher_PPF', 'pitcher_VORP', 'pitcher_FIP', 'pitcher_PVORP', 'pitcher_PWARP', 'pitcher_BASES_VS_AB', 'pitcher_hit_mult'])
df5 = pd.concat([df5,ones],axis=1)

for d in df5['Pitcher']:
    for i in ['pitcher_IP', 'pitcher_PA', 'pitcher_SO', 'pitcher_DRA', 'pitcher_ERA', 'pitcher_PPF', 'pitcher_VORP', 'pitcher_FIP', 'pitcher_PVORP', 'pitcher_PWARP', 'pitcher_BASES_VS_AB', 'pitcher_hit_mult']:
        df5[i][df5['Pitcher']==d] = float(game_logs2[i][game_logs2['Team_Pitch'] == d.upper()].values[0])
    


d_eff = ['def_H', 'def_HR',	'def_GB%', 'def_FB%', 'def_LD%', 'def_POP%', 'def_DP%']
ones = pd.DataFrame(np.ones((16,len(d_eff))), columns = ['def_H', 'def_HR',	'def_GB%', 'def_FB%', 'def_LD%', 'def_POP%', 'def_DP%'])
df5 = pd.concat([df5,ones],axis=1)

df5['team_ini'] = df5['def_H']

for tm in df5['Team']:
    df5['team_ini'][df5['Team']==tm] = game_logs2['Stadium'][game_logs2['team']==tm].values[0]


gl2 = game_logs2[['team_ini', 'def_H', 'def_HR', 'def_GB%', 'def_FB%', 'def_LD%', 'def_POP%', 'def_DP%']]

for tm in df5['team_ini']:
    for i in ['def_H', 'def_HR', 'def_GB%', 'def_FB%', 'def_LD%', 'def_POP%', 'def_DP%']:
        df5[i][df5['team_ini']==tm] = gl2[i][gl2['team_ini']==tm].values[0]

from constant_variables import TEAMS
def get_team_ini(team_name):
    for name, ini in TEAMS.items():
        if name == team_name:
            return ini
        
df5['Team_Bat_Ini'] = [get_team_ini(x.upper()) for x in df5['Team_Bat']]

bat_splits = ['bat_SO', 'bat_RBI', 'bat_DP', 'bat_FB%', 'bat_GB%', 'bat_LD%', 'bat_POP%', 'bat_ISO', 'bat_AVG', 'bat_OBP', 'bat_SLG', 'bat_TAV', 'batter_BASES_VS_AB',	'hit_mult']

ones = pd.DataFrame(np.ones((16,len(bat_splits))), columns = ['bat_SO', 'bat_RBI', 'bat_DP', 'bat_FB%', 'bat_GB%', 'bat_LD%', 'bat_POP%', 'bat_ISO', 'bat_AVG', 'bat_OBP', 'bat_SLG', 'bat_TAV', 'batter_BASES_VS_AB',	'hit_mult'])
df5 = pd.concat([df5,ones],axis=1)
'''
for tm,hd in zip(df5['Team_Bat_Ini'], df5['Throws']):
    for i in [ 'bat_SO', 'bat_RBI', 'bat_DP', 'bat_FB%', 'bat_GB%', 'bat_LD%', 'bat_POP%', 'bat_ISO', 'bat_AVG', 'bat_OBP', 'bat_SLG', 'bat_TAV', 'batter_BASES_VS_AB',	'hit_mult']:
        df5[i][df5['Team_Bat_Ini']==tm][df5['Throws']==hd] = round(float(game_logs2[i][game_logs2['Team_Bat']==tm][game_logs2['pitcher_hand']==hd].values.mean()),2)
'''
for tm,hd in zip(df5['Team_Bat_Ini'], df5['Throws']):
    for i in [ 'bat_SO', 'bat_RBI', 'bat_DP', 'bat_FB%', 'bat_GB%', 'bat_LD%', 'bat_POP%', 'bat_ISO', 'bat_AVG', 'bat_OBP', 'bat_SLG', 'bat_TAV', 'batter_BASES_VS_AB',	'hit_mult']:
        df5[i][df5['Team_Bat_Ini']==tm] = round(float(game_logs2[i][game_logs2['Team_Bat']==tm][game_logs2['pitcher_hand']==hd].values.mean()),2)



gl2 = game_logs2[['team_ini', 'def_H', 'def_HR', 'def_GB%', 'def_FB%', 'def_LD%', 'def_POP%', 'def_DP%']]

for tm in df5['team_ini']:
    for i in ['def_H', 'def_HR', 'def_GB%', 'def_FB%', 'def_LD%', 'def_POP%', 'def_DP%']:
        df5[i][df5['team_ini']==tm] = gl2[i][gl2['team_ini']==tm].values[0]


test, test2 = [], []
for tm,hd in zip(df5['Team_Bat_Ini'], df5['Throws']):
    for i in [ 'bat_SO', 'bat_RBI', 'bat_DP', 'bat_FB%', 'bat_GB%', 'bat_LD%', 'bat_POP%', 'bat_ISO', 'bat_AVG', 'bat_OBP', 'bat_SLG', 'bat_TAV', 'batter_BASES_VS_AB',	'hit_mult']:
       test.append( round(float(game_logs2[i][game_logs2['Team_Bat']==tm][game_logs2['pitcher_hand']==hd].values.mean()),2))
       test2.append([i,tm,hd])


bat_split = ['G_batter', 'PA_batter', 'R_batter', 'TB_batter', 'SO_batter', 'RBI_batter', 'DP_batter', 'FB%_batter', 'GB%_batter', 'LD%_batter', 'POP%_batter', 'ISO_batter', 'AVG_batter', 'OBP_batter', 'SLG_batter', 'TAV_batter', 'BASES_VS_AB_batter', 'SDTHB_BAT']
ones = pd.DataFrame(np.ones((16,len(bat_split))), columns = ['G_batter', 'PA_batter', 'R_batter', 'TB_batter', 'SO_batter', 'RBI_batter', 'DP_batter', 'FB%_batter', 'GB%_batter', 'LD%_batter', 'POP%_batter', 'ISO_batter', 'AVG_batter', 'OBP_batter', 'SLG_batter', 'TAV_batter', 'BASES_VS_AB_batter', 'SDTHB_BAT'])
df5 = pd.concat([df5,ones],axis=1)


for tm,hd in zip(df5['Team_Bat_Ini'], df5['Throws']):
    for i in ['G_batter', 'PA_batter', 'R_batter', 'TB_batter', 'SO_batter', 'RBI_batter', 'DP_batter', 'FB%_batter', 'GB%_batter', 'LD%_batter', 'POP%_batter', 'ISO_batter', 'AVG_batter', 'OBP_batter', 'SLG_batter', 'TAV_batter', 'BASES_VS_AB_batter', 'SDTHB_BAT']:
        df5[i][df5['Team_Bat_Ini']==tm] = round(float(game_logs2[i][game_logs2['Team_Bat']==tm][game_logs2['pitcher_hand']==hd].values[0]),2)

'''
for tm,hd in zip(df5['Team_Bat_Ini'], df5['Throws']):
    for i in ['G_batter', 'PA_batter', 'R_batter', 'TB_batter', 'SO_batter', 'RBI_batter', 'DP_batter', 'FB%_batter', 'GB%_batter', 'LD%_batter', 'POP%_batter', 'ISO_batter', 'AVG_batter', 'OBP_batter', 'SLG_batter', 'TAV_batter', 'BASES_VS_AB_batter', 'SDTHB_BAT']:
        df5[i][df5['Team_Bat_Ini']==tm][df5['Throws']==hd] = round(float(game_logs2[i][game_logs2['Team_Bat']==tm][game_logs2['pitcher_hand']==hd].values[0]),2)


batt, batt2 = [], []
for tm,hd in zip(df5['Team_Bat_Ini'], df5['Throws']):
    for i in ['G_batter', 'PA_batter', 'R_batter', 'TB_batter', 'SO_batter', 'RBI_batter', 'DP_batter', 'FB%_batter', 'GB%_batter', 'LD%_batter', 'POP%_batter', 'ISO_batter', 'AVG_batter', 'OBP_batter', 'SLG_batter', 'TAV_batter', 'BASES_VS_AB_batter', 'SDTHB_BAT']:
       batt.append( round(float(game_logs2[i][game_logs2['Team_Bat']==tm][game_logs2['pitcher_hand']==hd].values.mean()),2))
       batt2.append(i)
'''

game_logs2['team_ini'][game_logs2['Team_Pitch']=='MATT HARVEY'] = 'CIN'

side_pit_cols = ['H_SIDE_pit', '1B_SIDE_pit', '2B_SIDE_pit', '3B_SIDE_pit', 'HR_SIDE_pit', 'TB_SIDE_pit', 'SO_SIDE_pit', 'BB_SIDE_pit', 'HBP_SIDE_pit', 'SF_SIDE_pit', 'SH_SIDE_pit', 'DP_SIDE_pit', 'FB%_SIDE_pit', 'GB%_SIDE_pit', 'LD%_SIDE_pit', 'POP%_SIDE_pit', 'ISO_SIDE_pit', 'AVG_SIDE_pit', 'OBP_SIDE_pit', 'SLG_SIDE_pit', 'TAV_SIDE_pit', 'DRA_SIDE_pit', 'DRA-_SIDE_pit', 'CFIP_SIDE_pit', 'BASES_VS_AB_pit']
ones = pd.DataFrame(np.ones((16,len(side_pit_cols))), columns = side_pit_cols)
df5 = pd.concat([df5,ones],axis=1)

for tm,hd in zip(df5['Pitcher'], df5['team_ini']):
    for i in side_pit_cols:
        df5[i][df5['Pitcher']==tm] = round(float(game_logs2[i][game_logs2['Team_Pitch']==tm.upper()][game_logs2['team_ini']==hd].values[0]),2)



bullpen_cat = ['K/9', 'BB/9', 'K/BB', 'HR/9', 'K%', 'BB%', 'K-BB%', 'AVG', 'WHIP', 'BABIP', 'LOB%',
 'ERA-', 'FIP-', 'xFIP-', 'ERA', 'FIP', 'E-F', 'xFIP', 'SIERA']
ones = pd.DataFrame(np.ones((16,len(bullpen_cat))), columns = bullpen_cat)
df5 = pd.concat([df5,ones],axis=1)

for tm,hd in zip(df5['Pitcher'], df5['team_ini']):
    for i in bullpen_cat:
        df5[i][df5['Pitcher']==tm] = round(float(game_logs2[i][game_logs2['Team_Pitch']==tm.upper()][game_logs2['team_ini']==hd].values[0]),2)


monster_cols = ['Expected_Runs', 'Temp', 'Humidity', 'Rain']
ones = pd.DataFrame(np.ones((16,len(bat_split))), columns = monster_cols)
df5 = pd.concat([df5,ones],axis=1)

X_new = X[X.columns[list(X.columns).index('pitcher_SO'):]]
df5_new = df5[df5.columns[list(df5.columns).index('pitcher_SO'):]]

df5_new = df5_new.drop(['team_ini', 'Team_Bat_Ini', 'G_batter', 'PA_batter'], axis=1)


X_new.to_csv("C:/Users/mrcrb/PythonScripts/MLB/MLB/X_new.csv", sep=',', encoding='utf-8', index=False)
df5_new.to_csv("C:/Users/mrcrb/PythonScripts/MLB/MLB/df5_new.csv", sep=',', encoding='utf-8', index=False)







stats = {}
for i,j in zip(test2,test):
    if stats[i[0]] > 0:
        stats[i[0]]=[stats[i[0]],j]
    else:
        stats[i[0]]=j

i,j=0,0
test3, test4= [],[]
while j<len(test):
    if i<16:
        test3.append(test[j+i])
        i+=1
    else:
        test4.append(test3)
        j=j+i-1
        i=0
        test3=[]
i,j=0,0
batt3, batt4= [],[]
while j<len(batt):
    if i<16:
        batt3.append(batt[j+i])
        i+=1
    else:
        batt4.append(batt3)
        j=j+i-1
        i=0
        batt3=[]

test5 = pd.DataFrame([x for x in test4], columns = bat_splits)

p=0
stat = pd.DataFrame(pd.concat([pd.Series(p) for p in test4], axis=1), columns = bat_split)

'''from collections.abc import ChainMap
c = ChainMap()
class DeepChainMap(ChainMap):
    'Variant of ChainMap that allows direct updates to inner scopes'

    def __setitem__(self, key, value):
        for mapping in self.maps:
            if key in mapping:
                mapping[key] = value
                return
        self.maps[0][key] = value

    def __delitem__(self, key):
        for mapping in self.maps:
            if key in mapping:
                del mapping[key]
                return
        raise KeyError(key)

d = DeepChainMap(game_pit_bat_lineup_dic)'''




col_logs = ['Date',
 'Team_Bat',
 'Stadium',
 'Team_Pitch',
 'Expected_Runs',
 'Runs',
 'Temp',
 'Humidity',
 'Rain',
 'League',
 'player_id',
 'player_position',
 'team',
 'team_ini',
 'pitcher_hand',
 'pitcher_IP',
 'pitcher_PA',
 'pitcher_SO',
 'pitcher_DRA',
 'pitcher_ERA',
 'pitcher_PPF',
 'pitcher_VORP',
 'pitcher_FIP',
 'pitcher_PVORP',
 'pitcher_PWARP',
 'pitcher_BASES_VS_AB',
 'pitcher_hit_mult',
 'def_H',
 'def_HR',
 'def_GB%',
 'def_FB%',
 'def_LD%',
 'def_POP%',
 'def_DP%',
 'G_batter',
 'PA_batter',
 'R_batter',
 'TB_batter',
 'SO_batter',
 'RBI_batter',
 'DP_batter',
 'FB%_batter',
 'GB%_batter',
 'LD%_batter',
 'POP%_batter',
 'ISO_batter',
 'AVG_batter',
 'OBP_batter',
 'SLG_batter',
 'TAV_batter',
 'BASES_VS_AB_batter',
 'SDTHB_BAT',
 'bat_SO',
 'batBI',
 'bat_DP',
 'bat_FB%',
 'bat_GB%',
 'bat_LD%',
 'bat_POP%',
 'bat_ISO',
 'bat_AVG',
 'bat_OBP',
 'bat_SLG',
 'bat_TAV',
 'batter_BASES_VS_AB',
 'hit_mult']

#X1 = X.fillna(X.mean())
#X1 = pd.to_numeric(X1)

#df = df.groupby(df.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))

#game_logs.apply(lambda x: x.fillna(x.mean()),axis=0)
#X1.apply(lambda x: x.fillna(x.mean()),axis=0)

#game_logs['Expected_Runs'][287] = float(3.0)
#game_logs['Expected_Runs'] = game_logs['Expected_Runs'].astype(float)

