import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from mlb_classes1 import Players

df = pd.read_csv("MLB/game_scores.csv", encoding='latin-1')
df['INI'] = list(map((lambda x:x.split(' ')[1]), df['Date_Team_num']))

from constant_variables import TEAMS
def get_team_ini(team_name):
    for name, ini in TEAMS.items():
        if name == team_name:
            return ini

df['Stadium_Ini'] = list(map((lambda x:get_team_ini(x)), df['Stadium']))

game_dataframe = Players(data = df, 
                       column_names = ['Date_Team_num', 'Date', 'TEAM', 'INI', 'Opponent', 'Stadium', 
                                       'Stadium_Ini', 'Opponent Pitcher', 'Pitcher', 'Score', 'Game'],
                       column_titles = ['Date_Team', 'Date', 'Team', 'Ini', 'Opponent', 'Stadium', 
                                        'Stadium_Ini', 'Team_Pitcher', 'Opponent_Pitcher', 'Score', 'Game'])


# Create Instance from Class
game_frames = game_dataframe.create_df()

# Create Dictionary from Data
'''game_dic = game_dataframe.create_dict('Date')
game_dict_tot = game_dic[1]
game_dict_list = game_dic[2]'''

game_dic1 = game_dataframe.create_dict('Date_Team_num')
game_dict_tot = game_dic1[1]
game_dict_list = game_dic1[2]

# Create Dictionary of Left, Right and Switch Hitters
day1 = game_dataframe.select_players_by_attributes('2018-03-29', 'Date')
sox = game_dataframe.select_players_by_attributes('CHICAGO WHITE SOX', 'Team')
#switch = player_frame.select_players_by_attributes('S', 'bat_hand')

# Create Dictionary of Pitchers
pitchers = game_dataframe.select_players_by_attributes('CHRIS SALE', 'Team_Pitcher')


df1 = pd.read_csv("MLB/batter_all.csv", encoding='latin-1')
df1['mlb_team_long'] = list(map((lambda x:x.upper()), df1['mlb_team_long']))

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



# Add Pitcher Stats
df2 = pd.read_csv("MLB/pitcher_all.csv", encoding='latin-1')

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
    team_mean, team_info = {},{}
    for team in list(player_dic.keys()):
        for k,v in zip(list(player_dic[team].keys()[6:]), list(player_dic[team].iloc[:,6:].mean())):
            team_info[k] = v
        team_mean[team] = team_info
        team_info = {}
    return team_mean
    
team_mean = team_mean_batting(player_lineup)


game_pitcher_stats_dict = {}
for k,v in game_dict_list.items():
    for k1,v1 in v.items():
        pitcher = game_dict_list[k]['Opponent_Pitcher']
        opponent = game_dict_list[k]['Opponent']
        if pitcher in list(pitcher_dict_stats.keys()) and opponent == pitcher_dict_stats[pitcher]['team'].upper():
                game_pitcher_stats_dict[k] = {**game_dict_list[k], **pitcher_dict_stats[pitcher]}

from constant_variables import LEAGUE
def get_team_league(team_ini):
    for ini, league in LEAGUE.items():
        if ini == team_ini:
            return league



game_pitcher_batter_stats_dict, team_means = {},{}
team_mean_left = team_mean_batting(player_lineup_left)
team_mean_right = team_mean_batting(player_lineup_right)
team_mean_left_alt = team_mean_batting(player_lineup_left_alt)
team_mean_right_alt = team_mean_batting(player_lineup_right_alt)
for k,v in game_pitcher_stats_dict.items():
    for k1,v1 in v.items():
        team_ini = game_pitcher_stats_dict[k]['Ini']
        if game_pitcher_stats_dict[k]['pitcher_hand'] == 'L' and get_team_league(game_pitcher_stats_dict[k]['Ini']) == 'AL' and get_team_league(game_pitcher_stats_dict[k]['Stadium_Ini']) == 'AL':
            team_means[team_ini] = team_mean_left[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}
        elif game_pitcher_stats_dict[k]['pitcher_hand'] == 'R' and get_team_league(game_pitcher_stats_dict[k]['Ini']) == 'AL' and get_team_league(game_pitcher_stats_dict[k]['Stadium_Ini']) == 'AL':
            team_means[team_ini] = team_mean_right[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}
        elif game_pitcher_stats_dict[k]['pitcher_hand'] == 'L' and get_team_league(game_pitcher_stats_dict[k]['Ini']) == 'NL' and get_team_league(game_pitcher_stats_dict[k]['Stadium_Ini']) == 'AL':
            team_means[team_ini] = team_mean_left_alt[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}
        elif game_pitcher_stats_dict[k]['pitcher_hand'] == 'R' and get_team_league(game_pitcher_stats_dict[k]['Ini']) == 'AL' and get_team_league(game_pitcher_stats_dict[k]['Stadium_Ini']) == 'NL':
            team_means[team_ini] = team_mean_right_alt[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}
        elif game_pitcher_stats_dict[k]['pitcher_hand'] == 'L' and get_team_league(game_pitcher_stats_dict[k]['Ini']) == 'NL' and get_team_league(game_pitcher_stats_dict[k]['Stadium_Ini']) == 'NL':
            team_means[team_ini] = team_mean_left[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}
        elif game_pitcher_stats_dict[k]['pitcher_hand'] == 'R' and get_team_league(game_pitcher_stats_dict[k]['Ini']) == 'NL' and get_team_league(game_pitcher_stats_dict[k]['Stadium_Ini']) == 'NL':
            team_means[team_ini] = team_mean_right[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}
        elif game_pitcher_stats_dict[k]['pitcher_hand'] == 'L' and get_team_league(game_pitcher_stats_dict[k]['Ini']) == 'AL' and get_team_league(game_pitcher_stats_dict[k]['Stadium_Ini']) == 'NL':
            team_means[team_ini] = team_mean_left_alt[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}
        elif game_pitcher_stats_dict[k]['pitcher_hand'] == 'R' and get_team_league(game_pitcher_stats_dict[k]['Ini']) == 'NL' and get_team_league(game_pitcher_stats_dict[k]['Stadium_Ini']) == 'AL':
            team_means[team_ini] = team_mean_right_alt[team_ini]
            game_pitcher_batter_stats_dict[k] = {**game_pitcher_stats_dict[k], **team_means[team_ini]}


game_logs = pd.DataFrame(list(game_pitcher_batter_stats_dict.values()), columns = list(list(game_pitcher_batter_stats_dict.values())[0].keys()))
y = game_logs['Score']
X = game_logs.iloc[:,17:]


















# Create Dictionary of Left, Right and Switch Hitters
righty = player_frame.select_players_by_attributes('R', 'bat_hand')
lefty = player_frame.select_players_by_attributes('L', 'bat_hand')
switch = player_frame.select_players_by_attributes('S', 'bat_hand')

# Create Dictionary of Pitchers
pitchers = player_frame.select_players_by_attributes('P', 'player_position')





data = df

column_names = ['Date', 'TEAM', 'Opponent', 'Stadium', 
                'Opponent Pitcher', 'Pitcher', 'Score', 'Game', 'Date_Team_num']

column_titles = ['Date', 'Team', 'Opponent', 'Stadium', 
            'Team_Pitcher', 'Opponent_Pitcher', 'Score', 'Game', 'Date_Team']

player_col = 'Date_Team_num'


i=1
player_list, player_dict_list, player_dict_list_full = [],[],[]
player_dict, player_dict1, player_dictionary, dick = {},{},{},{}

for k,v in game_dict_tot.items():
    cat_len = len(v)
    for j,vs in enumerate(v):
        if list(vs.keys())[0] == 'Team_Pitcher':
            cat_num = j
    print(cat_len, cat_num)



list(vs.keys())[0]