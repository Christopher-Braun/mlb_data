import csv
import numpy as np
import pandas as pd
import re
import ast
from mlb_classes1 import Players

def date(date):
    date = date.replace('/','-')
    date = date.strip()
    if date[1] == '-':
        date = '0'+ date
    if date[4] == '-':
        date = date[:3] +'0'+ date[3:]
    return date

def daily_lineups_pitch():    
    df = pd.read_csv("MLB/game_date_info1.csv", encoding='latin-1')
    df1 = pd.read_csv("MLB/lineups_all.csv", encoding='latin-1')
    df['date'] = list(map((lambda x:date(x)), df['date']))
    #df1.columns = list(map((lambda x:date(x)), df1.columns.values))
    lineup_dict = {}
    for d,dc in zip(df1.loc[0], list(df1.columns)):
        lineup_dict[dc] = d
    
    lineup3_dict = {}    
    for k,v in lineup_dict.items():
        v = ast.literal_eval(v)
        lineup3_dict[k] = v
        
    dic_del, dic_key = [], []           
    for k,v in lineup3_dict.items():
        for k1,v1 in v.items():
            if len(v1) < 4:
                dic_del.append(k1)
                dic_key.append(k)
            
    for de, key in zip(dic_del, dic_key):
        lineup3_dict[key].pop(de, None)    
        
        
    for d in df['date'].unique():
        for k,v in lineup3_dict[d].items():
            if df[df['team_home']==k][df['date']==d].empty:
                league = list(df[df['team_away']==k][df['date']==d]['league'])[0]
                if league == 'NL':
                    lineup3_dict[d][k][list(df[df['team_away']==k][df['date']==d]['pitcher_away'])[0]] = 'SP'
            else:
                league = list(df[df['team_home']==k][df['date']==d]['league'])[0]
                if league == 'NL':
                    lineup3_dict[d][k][list(df[df['team_home']==k][df['date']==d]['pitcher_home'])[0]] = 'SP'
    return lineup3_dict   
