#! python3
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
import pandas as pd

# Importing the dataset
#workbook = xlrd.open_workbook('starting_pitchers.xlsx', on_demand = True)
#prev = pd.ExcelFile("starting_pitchers.xlsx")

df = pd.read_csv("MLB/starting_pitchers.csv", encoding='latin-1')

wiki = "https://www.baseball-reference.com/leagues/MLB/2018-schedule.shtml"
page = urllib.request.urlopen(wiki)
soup = BeautifulSoup(page, 'lxml')

print(soup.prettify())

# Scrape for teams and dates
i=0
teams, preview  = [], []
for i, row in enumerate(soup.find_all('a')):
    if (row.get_text()) == "Go to Today's Games":
        l = i
    elif (row.get_text()) == 'Minor Leagues':
        le = i
    teams.append(row.get_text())

teams = teams[l+1:le]

# Remove Boxscore
games = []
for team in teams:
    if team != 'Boxscore':
        games.append(team)
    
unplayed = games.index('Preview')

played = games[:unplayed-2]

day_index, day = [], []
for i,play in enumerate(played):
    if play[:4] == '2018':
        day_index.append(i)
        day.append(play)

# Determine games per day (schedule/2)
v, i = 0, 0
day_played, schedule = [], []
for i,j in enumerate(played):
    if i < day_index[v]:
        day_played.append(j)
    else:
        schedule.append(day_played)
        v+=1
        day_played = []

sche=0
for sch in schedule:
    sche += len(sch)

# Seperate Dates  
dates = []    
for i, j in zip(day, schedule):
    for k in range(len(j)):
        dates.append(i[:10])

# Seperate Teams
sched = []
for sc in schedule:
    sched = np.concatenate([sched,sc])

# Scrape for scores
x = []
for string in soup.strings:
    x.append(repr(string))

# Remove strings surrounding scores
hope = []
for xx in x:
    if "(" and ")" in xx:
        z = xx.split('(')
        w = z[1].split(")")
        hope.append(w[0])

# Game Scores
scores = []
for ho in hope:
    if len(ho) < 3 and ho.isdigit():
        scores.append(ho)
        
# Even and Odd indecies       
even, odd = [], []
for i in range(len(sched)):
    if i % 2 == 0:
        even.append(i)
    else:
        odd.append(i)
        
# Add Opponent team
oppo = []
for i,j in zip(even, odd):
    oppo.append(sched[j].upper())
    oppo.append(sched[i].upper())

# Shorten Date    
day_cut = list(map((lambda x:x[:10]), day))

# Add Stadium
i,j = 0,0
stadium = []
for i,j in zip(even, odd):
    stadium.append(oppo[i])
    stadium.append(oppo[i])

#ind = teams_uniq.index('LA Angels of Anaheim')
#teams_uniq[ind] = 'Los Angeles Angels'

df['TEAM'][df['TEAM']=='LA Angels of Anaheim'] = 'Los Angeles Angels'
df['TEAM'][df['TEAM']=="Oakland A's"] = 'Oakland Athletics'
df['TEAM'][df['TEAM']=="Arizona Diamondbacks"] = "Arizona D'Backs"


scores_team = pd.DataFrame(np.column_stack([dates, sched, scores, oppo, stadium]), columns = ['Date', 'Team', 'Score', 'Opponent', 'Stadium'])
team_list = pd.read_csv("MLB/team_list.csv", encoding='latin-1')
team_list['Team'][team_list['Team']=='Arizona Diamondbacks'] = "Arizona D'Backs"
scores_team = pd.merge(scores_team, team_list, on='Team', how='inner')
scores_team['Date_Team'] = scores_team['Date'] + ' ' + scores_team['TEAM']
team_list = team_list.rename(columns={'Team': 'Opponent'})
team_list = team_list.rename(columns={'TEAM': 'TEAM_DEFENSE'})
team_list['Opponent'] = list(map((lambda x:x.upper()),team_list['Opponent']))
scores_team1 = pd.merge(scores_team, team_list, on='Opponent', how='inner')
scores_team1['Date_Opponent'] = scores_team1['Date'] + ' ' + scores_team1['TEAM_DEFENSE']

scores_team1['Game'] = np.ones((len(scores_team1['Date_Team']),1))
j,i = 1,0
for date in list(scores_team1['Date_Team']):
    if list(scores_team1['Date_Team']).count(date) > 1:
        ind = scores_team1['Game'][scores_team1['Date_Team']==date].index.values
        for i in ind:
            scores_team1.loc[i, 'Game'] = j
            j+=1
    j=1
        

team_list = team_list.rename(columns={'Opponent': 'TEAM'})
df['TEAM'] = list(map((lambda x:x.upper()),df['TEAM']))
df1 = pd.merge(df, team_list, on='TEAM', how='inner')
df1['Date_Opponent'] = df1['DATE'] + ' ' + df1['TEAM_DEFENSE']

df1['Game'] = np.ones((len(df1['Date_Opponent']),1))
j,i = 1,0
for date in list(df1['Date_Opponent']):
    if list(df1['Date_Opponent']).count(date) > 1:
        ind = df1['Game'][df1['Date_Opponent']==date].index.values
        for i in ind:
            df1.loc[i, 'Game'] = j
            j+=1
    j=1

df1['Date_Opponent_num'] = df1['Date_Opponent'] + ' ' + df1['Game'].astype('str')
scores_team1['Date_Opponent_num'] = scores_team1['Date_Opponent'] + ' ' + scores_team1['Game'].astype('str')

scores_team2 = pd.merge(df1, scores_team1, on='Date_Opponent_num', how='inner')
df1 = df1.rename(columns={'Date_Opponent': 'Date_Team'})
df1['Date_Team_num'] = df1['Date_Team'] + ' ' + df1['Game'].astype('str')
scores_team2['Date_Team_num'] = scores_team2['Date_Team'] + ' ' + scores_team2['Game_y'].astype('str')

scores_team3 = pd.merge(df1, scores_team2, on='Date_Team_num', how='inner')

scores_team3_col = list(scores_team3.columns)
del_cols = ['DATE_x', 'TEAM_x', 'Date_Team_x', 'DATE_y', 'TEAM_y', 'TEAM_DEFENSE', 'TEAM_DEFENSE_x', 'Date_Opponent_x', 'Game_x', 'Date_Opponent_num_y', 'Date_Team_y', 'Date_Opponent_y', 'Game_y']
scores_team4 = scores_team3.drop(del_cols, axis=1)

scores_team4 = scores_team4.rename(columns={'PITCHER_x': 'Opponent Pitcher'})
scores_team4 = scores_team4.rename(columns={'PITCHER_y': 'Pitcher'})
scores_team4['Opponent Pitcher'] = list(map((lambda x:x.upper()),scores_team4['Opponent Pitcher']))
scores_team4['Pitcher'] = list(map((lambda x:x.upper()),scores_team4['Pitcher']))

for pitcher in scores_team4['Pitcher']:
    if pitcher == 'MATT KOCH':
        scores_team4['Pitcher'][scores_team4['Pitcher']=='MATT KOCH'] = 'MATTHEW KOCH'
    if pitcher == 'JAKE FARIA':
        scores_team4['Pitcher'][scores_team4['Pitcher']=='JAKE FARIA'] = 'JACOB FARIA'
    if pitcher == 'JAKOB JUNIS':
        scores_team4['Pitcher'][scores_team4['Pitcher']=='JAKOB JUNIS'] = 'JAKE JUNIS'
    if pitcher == 'JAMIE GARCIA':
        scores_team4['Pitcher'][scores_team4['Pitcher']=='JAMIE GARCIA'] = 'JAIME GARCIA'


'''
teams_uniq = list(df['TEAM'].unique())

dft = pd.DataFrame(np.ones([len(teams_uniq),len(day_cut)], dtype='str'), index =teams_uniq, columns = day_cut )
i,j,k=0,0,0
games_byday = []
for k,block in enumerate(schedule):
    i = len(block)
    #print(i,j,k)
    for teamu in block:
        dft.loc[teamu][k] = np.array2string(df['PITCHER'][j:i+j][df['TEAM'][j:i+j]==teamu].values)[2:-2]
        #print(j,k,i)
    j += i
    #print(k,i,j)

i,j=0,0
d = []
pit_ser = pd.Series(np.ones(len(scores_team['Team']), dtype='str'))
for k,team_s in enumerate(scores_team['Team']):
    ind = scores_team['Team'][scores_team['Team']==team_s].index
    for i,j in enumerate(ind):
        for d_f in dft.loc[team_s]:
            if d_f != '1' and len(d_f.split("' '"))==1:
                d.append(d_f)
                pit_ser.loc[j] = d[i]
            elif d_f != '1' and len(d_f.split("' '"))==2:
                g = d_f.split("' '")
                d.append(g[0])
                d.append(g[1])
                pit_ser.loc[j] = d[i]
                
    d = []
                

        
# Add Opponent Pitcher
i,j = 0,0
opp_pit = []
for i,j in zip(even, odd):
    opp_pit.append(pit_ser[j])
    opp_pit.append(pit_ser[i])
    
# Add Opponent Pitcher
i,j = 0,0
opp_pit = []
for i,j in zip(even, odd):
    opp_pit.append(sched[j])
    opp_pit.append(sched[i])
    
    
opp_up = []
for op in opp_pit:
    if op is None:
        up = 'PPD'
        opp_up.append(up)
    else:
        up = op.upper()
        opp_up.append(up)
'''
        
pitching_total = scores_team4



#pitching_total['Home Pitcher'] = pit_ser
#pitching_total['Opponent'] = opp_up

    
#pitching_total['Stadium'] = stadium

#pitching_total['Home Pitcher'] = list(map((lambda x:x.strip()),pitching_total['Home Pitcher']))
#pitching_total['Opponent Pitcher'] = list(map((lambda x:x.strip()),pitching_total['Opponent Pitcher']))
 

scoring = pitching_total
scoring.to_csv('MLB/game_scores.csv', sep=',', encoding='utf-8')

