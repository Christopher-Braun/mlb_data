import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict

df = pd.read_csv("MLB/batter_all.csv", encoding='latin-1')

player_dict = dict(zip(df['mlb_name'], df['mlb_pos']))

players = {'player_name': df['mlb_name'],
           'player_position': df['mlb_pos'],
           'team': df['mlb_team_long']}

df['mlb_team_long'] = list(map((lambda x:x.upper()), df['mlb_team_long']))


class Players(object):
    """Basic Player Information - Name, Team, Position, Hand...
    """
    #players_df = pd.read_csv("MLB/batter_all.csv", encoding='latin-1')
    
    def __init__(self, data, column_titles, column_names):
        self.data = data
        self.column_titles = column_titles
        self.column_names = column_names
        
    def create_df(self):
        self.dataframe = pd.DataFrame(np.column_stack([[x for x in self.data[self.column_names].values]]),
                                      columns = self.column_titles)
        return self.dataframe
    
    def create_dict(self, player_col):
        i=1
        player_list, player_dict_list, player_dict_list_full = [],[],[]
        player_dict, player_dict1, player_dictionary, dick = {},{},{},{}
        for player in self.data[player_col]:
            while i <= len(self.column_names):
                if i < len(self.column_names) and i > 1:
                    player_dict = {self.column_titles[i]: list(self.data[self.column_names[i]][self.data['mlb_name'] == player].astype('str'))[0]}
                    player_list.append(player_dict)
                    player_dict1[self.column_titles[i]] = list(self.data[self.column_names[i]][self.data['mlb_name'] == player].astype('str'))[0]
                    i+=1
                if i == 1:
                    player_dict = {self.column_titles[i]: list(self.data[self.column_names[i]][self.data['mlb_name'] == player].astype('str'))[0]}
                    player_list.append(player_dict)
                    player_dict1[self.column_titles[i]] = list(self.data[self.column_names[i]][self.data['mlb_name'] == player].astype('str'))[0]
                    i+=1
                if i == len(self.column_names):
                    #player_dict_list_full would be good for Team seperation
                    player_dict_list_full.append(player_dict1)
                    dick[player] = player_dict1
                    player_dictionary[player] = player_list
                    player_list = []
                    player_dict1 = {}
                    i+=1
            i=1
            
        self.dick = dick
        self.player_dic_lib = player_dictionary
        return player_dict_list_full, player_dictionary, dick    
    
    def right_handed_batters(self):
        right_handed = {}
        right, hand = [], []
        for k, [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15] in self.dick.items():
            if self.dick[k][v4] == 'R':
                right.append(k)
                hand.append(self.dick[k][v4])
                right_handed[k] = {v1:self.dick[k][v1], 
                            v2:self.dick[k][v2], 
                            v3:self.dick[k][v3], 
                            v4:self.dick[k][v4], 
                            v5:self.dick[k][v5],
                            v6:self.dick[k][v6], 
                            v7:self.dick[k][v7], 
                            v8:self.dick[k][v8], 
                            v9:self.dick[k][v9], 
                            v10:self.dick[k][v10],
                            v11:self.dick[k][v11], 
                            v12:self.dick[k][v12], 
                            v13:self.dick[k][v13], 
                            v14:self.dick[k][v14], 
                            v15:self.dick[k][v15]}
        
        self.right_handed = right_handed
        self.right_batters = right
        return right_handed
    
    def left_handed_batters(self):
        left_handed = {}
        left, hand = [], []
        for k, [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15] in self.dick.items():
            if self.dick[k][v4] == 'L':
                left.append(k)
                hand.append(self.dick[k][v4])
                left_handed[k] = {v1:self.dick[k][v1], 
                           v2:self.dick[k][v2], 
                           v3:self.dick[k][v3], 
                           v4:self.dick[k][v4], 
                           v5:self.dick[k][v5],
                           v6:self.dick[k][v6], 
                           v7:self.dick[k][v7], 
                           v8:self.dick[k][v8], 
                           v9:self.dick[k][v9], 
                           v10:self.dick[k][v10],
                           v11:self.dick[k][v11], 
                           v12:self.dick[k][v12], 
                           v13:self.dick[k][v13], 
                           v14:self.dick[k][v14], 
                           v15:self.dick[k][v15]}
        
        self.left_handed = left_handed
        self.left_batters = left
        return left_handed

    def switch_handed_batters(self):
        switch_handed = {}
        switch, hand = [], []
        for k, [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15] in self.dick.items():
            if self.dick[k][v4] == 'S':
                switch.append(k)
                hand.append(self.dick[k][v4])
                switch_handed[k] = {v1:self.dick[k][v1], 
                             v2:self.dick[k][v2], 
                             v3:self.dick[k][v3], 
                             v4:self.dick[k][v4], 
                             v5:self.dick[k][v5],
                             v6:self.dick[k][v6], 
                             v7:self.dick[k][v7], 
                             v8:self.dick[k][v8], 
                             v9:self.dick[k][v9], 
                             v10:self.dick[k][v10],
                             v11:self.dick[k][v11], 
                             v12:self.dick[k][v12], 
                             v13:self.dick[k][v13], 
                             v14:self.dick[k][v14], 
                             v15:self.dick[k][v15]}
            
        self.switch_handed = switch_handed
        self.switch_batters = switch
        return switch_handed
    
    def right_handed_pitchers(self, cat_name, cat_value):
        right_handed_pitcher = {}
        right, hand = [], []
        cat1, cat2 = self.find_length_cat(cat_name)
        vars = self.create_dict_vars(cat1)
        for k, vars in self.dick.items():
            if self.dick[k][cat_value] == cat_name:
                right.append(k)
                hand.append(self.dick[k][cat_value])
                right_handed_pitcher[k] = vars
        
        self.right_handed_pitcher = right_handed_pitcher
        self.right_pitchers = right
        return right_handed_pitcher

            
    def find_length_cat(self, cat_name):
        for k,v in self.player_dic_lib.items():
            cat_len = len(v)
            for j,vs in enumerate(v):
                if list(vs.values())[0] == cat_name:
                    cat_num = j
        return cat_len, cat_num
    
    def create_dict_vars(self, cat_len):
        vars = []
        for i in range(1,cat_len+1):
            vars.append('v' + str(i))
        return vars
        
            
            
            
            
            
player_frame = Players(data = df, column_names = ['mlb_name', 'mlb_pos', 'mlb_team_long', 'TEAM', 'bats', 'LG_batter_Std',
                                                  'OBP_batter_Std', 'SLG_batter_Std', 'OPS_batter_Std', 'ISO_batter_Std',
                                                  'oppOPS_batter_Std', 'TAv_batter_Std', 'VORP_batter_Std', 'FRAA_batter_Std', 
                                                  'BWARP_batter_Std', 'BASES_VS_AB_batter_Std'],
                               column_titles = ['player_name', 'player_position', 'team', 'team_ini', 'bat_hand', 'league',
                                                'batter_OBP', 'batter_SLG', 'batter_OPS', 'batter_ISO', 'batter_oppOPS',
                                                'batter_TAv', 'batter_VORP', 'batter_FRAA', 'batter_BWARP', 'batter_BASES_VS_AB'])


player_frames = player_frame.create_df()
player_dic = player_frame.create_dict('mlb_name')

player_dict_tot_list = player_dic[0]
player_dict_tot = player_dic[1]
player_dict_stats = player_dic[2]

righty = player_frame.right_handed_batters()
lefty = player_frame.left_handed_batters()
switch = player_frame.switch_handed_batters()

pitchers = player_frame.right_handed_pitchers('P', 'player_position')

test = player_frame.find_length_cat('P')
test1 = player_frame.create_dict_vars(15)

            