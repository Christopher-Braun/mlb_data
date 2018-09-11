import numpy as np
import pandas as pd

df = pd.read_csv("MLB/bullpen_stats.csv", encoding='latin-1')

from constant_variables import TEAMS_SHORT
def get_team_ini(team_name):
    for name, ini in TEAMS_SHORT.items():
        if name == team_name:
            return ini
        
df['Team_Ini'] = list(map((lambda x:get_team_ini(x)), df['Team']))
      

cats = ['K%', 'BB%', 'K-BB%', 'LOB%']
[float(d[:-1]) for cat in cats for d in df[cat]]
        
for cat in cats:
    df[cat] = [float(d[:-1]) for d in df[cat]] 
        
df.to_csv("C:/Users/mrcrb/PythonScripts/MLB/MLB/bullpen.csv", sep=',', encoding='utf-8', index=False)

        
        
        
        
        