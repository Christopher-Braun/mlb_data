import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict
from constant_variables import TEAMS

'''df = pd.read_csv("MLB/batter_all.csv", encoding='latin-1')

player_dict = dict(zip(df['mlb_name'], df['mlb_pos']))

players = {'player_name': df['mlb_name'],
           'player_position': df['mlb_pos'],
           'team': df['mlb_team_long']}

df['mlb_team_long'] = list(map((lambda x:x.upper()), df['mlb_team_long']))'''


class Players(object):
    """Basic Player Information - Name, Team, Position, Hand...
    """
    #players_df = pd.read_csv("MLB/batter_all.csv", encoding='latin-1')
    
    def __init__(self, data, column_titles, column_names):
        self.data = data
        self.column_titles = column_titles
        self.column_names = column_names
        self.column_names_left = ['mlb_name', 'mlb_pos', 'mlb_team_long', 'TEAM', 'bats', 'LG_batter_Std',
                                             'G_LEFT_batter_Std', 'PA_LEFT_batter_Std', 'R_LEFT_batter_Std', 'TB_LEFT_batter_Std', 
                                             'SO_LEFT_batter_Std', 'RBI_LEFT_batter_Std', 'DP_LEFT_batter_Std', 'FB%_LEFT_batter_Std', 
                                             'GB%_LEFT_batter_Std', 'LD%_LEFT_batter_Std', 'POP%_LEFT_batter_Std', 'ISO_LEFT_batter_Std', 
                                             'AVG_LEFT_batter_Std', 'OBP_LEFT_batter_Std', 'SLG_LEFT_batter_Std', 'TAV_LEFT_batter_Std', 
                                             'BASES_VS_AB_LEFT_batter_Std', 'SDTHB_BATR']
        self.column_names_right = ['mlb_name', 'mlb_pos', 'mlb_team_long', 'TEAM', 'bats', 'LG_batter_Std',
                                             'G_RIGHTbatter_Std', 'PA_RIGHTbatter_Std', 'R_RIGHTbatter_Std', 'TB_RIGHTbatter_Std', 
                                             'SO_RIGHTbatter_Std', 'RBI_RIGHTbatter_Std', 'DP_RIGHTbatter_Std', 'FB%_RIGHTbatter_Std', 
                                             'GB%_RIGHTbatter_Std', 'LD%_RIGHTbatter_Std', 'POP%_RIGHTbatter_Std', 'ISO_RIGHTbatter_Std', 
                                             'AVG_RIGHTbatter_Std', 'OBP_RIGHTbatter_Std', 'SLG_RIGHTbatter_Std', 'TAV_RIGHTbatter_Std', 
                                             'BASES_VS_AB_RIGHTbatter_Std', 'SDTHB_BATR']
        self.column_names_all = ['mlb_name', 'mlb_id', 'mlb_pos', 'mlb_team_long', 'TEAM', 'bats', 'LG_batter_Std',
                                                  'PA_batter_Std', 'R_batter_Std', 'SO_batter_Std', 'RBI_batter_Std', 'DP_batter_Std', 'AVG_batter_Std', 'OBP_batter_Std', 'SLG_batter_Std', 'OPS_batter_Std', 'ISO_batter_Std',
                                                  'oppOPS_batter_Std', 'TAv_batter_Std', 'VORP_batter_Std', 'FRAA_batter_Std', 
                                                  'BWARP_batter_Std', 'BASES_VS_AB_batter_Std', 'SDTHB_BAT']
        
    def create_df(self):
        self.dataframe = pd.DataFrame(np.column_stack([[x for x in self.data[self.column_names].values]]),
                                      columns = self.column_titles)
        return self.dataframe
    
    def create_dict(self, player_col):
        i=1
        player_list, player_dict_list = [],[]
        player_dict, player_dict1, player_dictionary, dick = {},{},{},{}
        for player in self.data[player_col]:
            while i <= len(self.column_names):
                if i < len(self.column_names) and i > 1:
                    player_dict = {self.column_titles[i]: list(self.data[self.column_names[i]][self.data[player_col] == player].astype('str'))[0]}
                    player_list.append(player_dict)
                    player_dict1[self.column_titles[i]] = list(self.data[self.column_names[i]][self.data[player_col] == player].astype('str'))[0]
                    i+=1
                if i == 1:
                    player_dict = {self.column_titles[i]: list(self.data[self.column_names[i]][self.data[player_col] == player].astype('str'))[0]}
                    player_list.append(player_dict)
                    player_dict1[self.column_titles[i]] = list(self.data[self.column_names[i]][self.data[player_col] == player].astype('str'))[0]
                    i+=1
                if i == len(self.column_names):
                    #player_dict_list_full would be good for Team seperation
                    #player_dict_list_full.append(player_dict1)
                    dick[player] = player_dict1
                    player_dictionary[player] = player_list
                    player_list = []
                    player_dict1 = {}
                    i+=1
            i=1
            
        self.dick = dick
        self.player_dic_lib = player_dictionary
        return player_dict_list, player_dictionary, dick
    
    def select_players_by_attributes(self, cat_name, cat_value):
        right_handed_pitcher = {}
        right, hand = [], []
        cat1, cat2 = self.find_length_cat(cat_value)
        vars = self.create_dict_vars(cat1)
        for k, vars in self.dick.items():
            if self.dick[k][cat_value] == cat_name:
                right.append(k)
                hand.append(self.dick[k][cat_value])
                right_handed_pitcher[k] = vars
        
        self.right_handed_pitcher = right_handed_pitcher
        self.right_pitchers = right
        return right_handed_pitcher

            
    def find_length_cat(self, cat_value):
        for k,v in self.player_dic_lib.items():
            cat_len = len(v)
            for j,vs in enumerate(v):
                if list(vs.keys())[0] == cat_value:
                    cat_num = j
        return cat_len, cat_num
    
    #vars = ['v' + str(i) for i in range(1,cat_len+1)]
    def create_dict_vars(self, cat_len):
        vars = []
        for i in range(1,cat_len+1):
            vars.append('v' + str(i))
        return vars
        
    def create_lineups(self, cat_all, cat_right, cat_left):
        team_names = list(self.data['TEAM'].unique())
        team_lineup, team_lineup_right, team_lineup_left, team_lineup_right_alt, team_lineup_left_alt = {}, {}, {}, {}, {}
        pitch = self.data[self.data['mlb_pos'] == 'P'][self.data['PA_batter_Std'] > 5]
        pitch_mean = list(pitch.iloc[:,10:].mean())
        pitch_mean[:10] = list(map((lambda x:int(x)),pitch_mean[:10]))
        pitch_mean[34:48] = list(map((lambda x:int(x)),pitch_mean[34:48]))
        pitch_mean[60:74] = list(map((lambda x:int(x)),pitch_mean[60:74]))
        pp = ['P','P','P','P','P','P','P','P','P', 'P']
        pitch_mean1 = pd.DataFrame([pp + pitch_mean], columns = pitch.columns)

        for team in team_names:
            if self.data[self.data['TEAM']==team][self.data['LG_batter_Std'] == 'AL'].empty:
                data_team = self.data[self.data['TEAM']==team].sort_values(['PA_batter_Std'], ascending = False)
                data_team_right = self.data[self.data['TEAM']==team].sort_values(['PA_RIGHTbatter_Std'], ascending = False)
                data_team_left = self.data[self.data['TEAM']==team].sort_values(['PA_LEFT_batter_Std'], ascending = False)
                team_lineup[team] = pd.concat([data_team.iloc[0:9,cat_all], pitch_mean1.iloc[:,cat_all]])
                team_lineup_right[team] = pd.concat([data_team_right.iloc[0:9, cat_right], pitch_mean1.iloc[:,cat_right]])
                team_lineup_left[team] = pd.concat([data_team_left.iloc[0:9,cat_left], pitch_mean1.iloc[:,cat_left]])
                team_lineup_right_alt[team] = data_team_right.iloc[0:10,cat_right]
                team_lineup_left_alt[team] = data_team_left.iloc[0:10,cat_left]
            elif self.data[self.data['TEAM']==team][self.data['LG_batter_Std'] == 'NL'].empty:
                data_team = self.data[self.data['TEAM']==team].sort_values(['PA_batter_Std'], ascending = False)
                data_team_right = self.data[self.data['TEAM']==team].sort_values(['PA_RIGHTbatter_Std'], ascending = False)
                data_team_left = self.data[self.data['TEAM']==team].sort_values(['PA_LEFT_batter_Std'], ascending = False)
                team_lineup[team] = data_team.iloc[0:10,cat_all]
                team_lineup_right[team] = data_team_right.iloc[0:10,cat_right]
                team_lineup_left[team] = data_team_left.iloc[0:10,cat_left]
                team_lineup_right_alt[team] = pd.concat([data_team_right.iloc[0:9, cat_right], pitch_mean1.iloc[:,cat_right]])
                team_lineup_left_alt[team] = pd.concat([data_team_left.iloc[0:9,cat_left], pitch_mean1.iloc[:,cat_left]])
        return team_lineup, team_lineup_right, team_lineup_left, team_lineup_right_alt, team_lineup_left_alt
            
    
    
            
            
'''            
player_frame = Players(data = df, 
                       column_names = ['mlb_name', 'mlb_pos', 'mlb_team_long', 'TEAM', 'bats', 'LG_batter_Std',
                                                  'OBP_batter_Std', 'SLG_batter_Std', 'OPS_batter_Std', 'ISO_batter_Std',
                                                  'oppOPS_batter_Std', 'TAv_batter_Std', 'VORP_batter_Std', 'FRAA_batter_Std', 
                                                  'BWARP_batter_Std', 'BASES_VS_AB_batter_Std'],
                       column_titles = ['player_name', 'player_position', 'team', 'team_ini', 'bat_hand', 'league',
                                        'batter_OBP', 'batter_SLG', 'batter_OPS', 'batter_ISO', 'batter_oppOPS',
                                        'batter_TAv', 'batter_VORP', 'batter_FRAA', 'batter_BWARP', 'batter_BASES_VS_AB'])

# Create Instance from Class
player_frames = player_frame.create_df()

# Create Dictionary from Data
player_dic = player_frame.create_dict('mlb_name')
player_dict_tot = player_dic[1]
player_dict_stats = player_dic[2]

# Create Dictionary of Left, Right and Switch Hitters
righty = player_frame.select_players_by_attributes('R', 'bat_hand')
lefty = player_frame.select_players_by_attributes('L', 'bat_hand')
switch = player_frame.select_players_by_attributes('S', 'bat_hand')

# Create Dictionary of Pitchers
pitchers = player_frame.select_players_by_attributes('P', 'player_position')

# Testing function (can delete)
test = player_frame.find_length_cat('P')
test1 = player_frame.create_dict_vars(15)'''

def create_lineups(self):
    team_names = list(self.data['TEAM'].unique())
    team_lineup, team_lineup_right, team_lineup_left = {}, {}, {}
    for team in team_names:
        if self.data[self.data['TEAM']==team][self.data['LG_batter_Std'] == 'AL'].empty:
            data_team = self.data[self.data['TEAM']==team].sort_values(['PA_batter_Std'], ascending = False)
            data_team_right = self.data[self.data['TEAM']==team].sort_values(['PA_RIGHTbatter_Std'], ascending = False)
            data_team_left = self.data[self.data['TEAM']==team].sort_values(['PA_LEFT_batter_Std'], ascending = False)
            team_lineup[team] = data_team.iloc[0:9,:]
            team_lineup_right[team] = data_team_right.iloc[0:9,:]
            team_lineup_left[team] = data_team_left.iloc[0:9,:]
        elif self.data[self.data['TEAM']==team][self.data['LG_batter_Std'] == 'NL'].empty:
            data_team = self.data[self.data['TEAM']==team].sort_values(['PA_batter_Std'], ascending = False)
            data_team_right = self.data[self.data['TEAM']==team].sort_values(['PA_RIGHTbatter_Std'], ascending = False)
            data_team_left = self.data[self.data['TEAM']==team].sort_values(['PA_LEFT_batter_Std'], ascending = False)
            team_lineup[team] = data_team.iloc[0:10,:]
            team_lineup_right[team] = data_team_right.iloc[0:10,:]
            team_lineup_left[team] = data_team_left.iloc[0:10,:]
    return team_lineup, team_lineup_right, team_lineup_left      