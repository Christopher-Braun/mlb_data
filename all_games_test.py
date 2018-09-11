#! python3
from selenium import webdriver
from bs4 import BeautifulSoup
import urllib.request
import numpy as np
import pandas as pd
import time
import csv
from open_dict import daily_lineups_pitch

games_total = pd.DataFrame(np.ones((1,12)), columns = ['date', 'team_away', 'team_home', 'pitcher_away', 'pitcher_home', 'expected_runs_away', 'actual_runs_away', 'expected_runs_home', 'actual_runs_home', 'temp', 'humidity', 'rain'])
from constant_variables import TEAM_INITIALS
def get_team_ini(team_ini):
    for name, ini in TEAM_INITIALS.items():
        if ini == team_ini:
            return name
        
from constant_variables import LEAGUE
def get_team_league(team_ini):
    for ini, league in LEAGUE.items():
        if ini == team_ini:
            return league
'''        
def daily_lineups_pitch(df_saved):    
    #df = pd.read_csv("MLB/game_date_info1.csv", encoding='latin-1')
    #df1 = pd.read_csv("MLB/lineups_all.csv", encoding='latin-1')
    lineup_dict = {}
    for d,dc in zip(df_saved.loc[1], list(df_saved.columns)):
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
    return lineup3_dict
'''

# Change date
driver = webdriver.Chrome()
driver.get("https://baseballmonster.com/boxscores.aspx")
elem1 = driver.find_element_by_name("BACK")
elem1.click()
time.sleep(20)
soup = BeautifulSoup(driver.page_source, 'lxml')

def separate_games(soup, games_total=None, all_lineups=None, scraped=0):
    # Get Starting Pitchers
    strangs = []
    for string in soup.strings:
        strangs.append(repr(string))
    
    starting_pitchers, order, pitch_team = [], [], []
    for i,line in enumerate(strangs):
        if line == "'SP'":
            starting_pitchers.append(strangs[i-1][1:-1])
            order.append(strangs[i+3][1:-1])
            pitch_team.append(strangs[i+2][1:-1])

    starters, index = [],[]
    i=0
    for aw, hm in zip(starting_pitchers, order):
        if hm == '1':
            starters.append(aw)
            i+=1
        else:
            index.append(i)
            i+=1
            
    if len(index)>0:
        for ind in index: 
            pitch_team.pop(ind)
            
    # Get game info
    team_away, team_home, expected_runs_away, actual_runs_away, expected_runs_home, actual_runs_home, weather = [], [], [], [], [], [], []
    for i,row in enumerate(soup.find_all('td')):
        if row.get_text() == 'View':
            team_away.append(soup.find_all('td')[i+1].get_text())
            team_home.append(soup.find_all('td')[i+4].get_text()[2:])
            expected_runs_away.append(soup.find_all('td')[i+2].get_text().strip())
            actual_runs_away.append(soup.find_all('td')[i+3].get_text().strip())
            expected_runs_home.append(soup.find_all('td')[i+5].get_text().strip())
            actual_runs_home.append(soup.find_all('td')[i+6].get_text().strip())
            weather.append(soup.find_all('td')[i+9].get_text().strip())
            
    # Get and Format Date
    for table in soup.select('h1'):
         date = table.get_text()[-9:]
    
    date = date.replace('/', '-')
    date = date.strip()
    if date[1] == '-':
        date = '0'+ date
    if date[4] == '-':
        date = date[:3] +'0'+ date[3:]
    dates = []
    for i in team_away:
        dates.append(date)
        
            
    # for games before trade (need to add a date arguement)        
    if 'Matt Harvey' in starters and date == '04-19-2018' or date == '04-14-2018' or date == '04-08-2018' or date == '04-03-2018':
        mh_ind = starters.index('Matt Harvey')
        pitch_team[mh_ind] = 'NYM'
    if 'Buddy Baumann' in starters and date == '04-11-2018':
        mh_ind = starters.index('Buddy Baumann')
        pitch_team[mh_ind] = 'SD'
    if 'A.J. Cole' in starters and date == '04-11-2018' or date == '04-03-2018':
        mh_ind = starters.index('A.J. Cole')
        pitch_team[mh_ind] = 'WAS'
    if 'Chris Archer' in starters and int(date[1]) >= 8:
        mh_ind = starters.index('Chris Archer')
        pitch_team[mh_ind] = 'PIT'
    if 'Tyler Clippard' in starters and date == '08-02-2018':
        mh_ind = starters.index('Tyler Clippard')
        starters[mh_ind] = 'Mike Hauschild'
    if 'Sonny Gray' in starters and int(date[1]) >= 7 and int(date[3:5]) > 22 or 'Sonny Gray' in starters and int(date[1]) >= 8:
        mh_ind = starters.index('Sonny Gray')
        pitch_team[mh_ind] = 'NYY'
    if 'Cole Hamels' in starters and int(date[1]) >= 7 and int(date[3:5]) > 26 or 'Cole Hamels' in starters and int(date[1]) >= 8:
        mh_ind = starters.index('Cole Hamels')
        pitch_team[mh_ind] = 'CHC'      
    if 'Lance Lynn' in starters and int(date[1]) >= 8:
        mh_ind = starters.index('Lance Lynn')
        pitch_team[mh_ind] = 'NYY'
    '''if 'Oliver Drake' in starters:
        mh_ind = starters.index('Oliver Drake')
        pitch_team[mh_ind] = 'MIL'    
    if 'Enny Romero' in starters:
        mh_ind = starters.index('Enny Romero')
        pitch_team[mh_ind] = 'WAS' '''


    
    start_index = []
    for i,st in enumerate(starters):
        for j,tm in enumerate(pitch_team):
            if tm != pitch_team[i-1]:
                start_index.append(i)
                #starters.pop(i)
                #pitch_team.pop(j)
                
    start_index, starting, team_list = [], [], []
    i,j = 0,0
    for st,tm in zip(starters, pitch_team):
            if tm != pitch_team[i-1]:
                start_index.append(i)
                starting.append(st)
                team_list.append(tm)
                i+=1
            else:
                i+=1
    
    
    # Check if SP was listed for each game (Harvey Team wrong - replace 16 with 36)
    total, total_teams = [], []
    for i,row in enumerate(soup.find_all('td')):
        if row.get_text() == 'Totals':
            if soup.find_all('td')[i-19].get_text() == 'Oliver Drake':
                print('BOOOYAAA')
                total.append(soup.find_all('td')[i+5].get_text())
                total_teams.append('MIL')
            elif soup.find_all('td')[i-19].get_text() == 'Matt Harvey':
                total.append(soup.find_all('td')[i+5].get_text())
                total_teams.append('NYM')
            elif soup.find_all('td')[i-19].get_text() == 'Buddy Baumann':
                total.append(soup.find_all('td')[i+5].get_text())
                total_teams.append('SD')                
            elif soup.find_all('td')[i-19].get_text() == 'Wilmer Font':
                total.append(soup.find_all('td')[i+5].get_text())
                total_teams.append('LAD')                    
            elif soup.find_all('td')[i-19].get_text() == 'Enny Romero':
                total.append(soup.find_all('td')[i+5].get_text())
                total_teams.append('WAS')
            else:
                print(soup.find_all('td')[i-19].get_text())
                total.append(soup.find_all('td')[i+5].get_text())
                total_teams.append(soup.find_all('td')[i-16].get_text())
                
                
                
    totals_away, totals_home, away_team, home_team, teams = [], [], [], [], []
    for i,tot in enumerate(total):
        if (i+1)%4 == 0 and i != 0:
            totals_home.append(tot)
            totals_away.append(total[i-1])
            home_team.append(total_teams[i])
            away_team.append(total_teams[i-1])
            teams.append(total_teams[i])
            teams.append(total_teams[i-1])
            
    pitch_add, pitch_index = [], []     
    for i,team in enumerate(teams):
        if team not in team_list:
            pitch_add.append(team)
            pitch_index.append(i)
            starting.insert(i, 'Unknown')
            
    pitcher_away, pitcher_home, team_a, team_h = [], [], [], []
    for i,j in enumerate(starting):
        if i%2 == 0:
            pitcher_away.append(j.upper())
            team_a.append(teams[i])
        else:
            pitcher_home.append(j.upper())
            team_h.append(teams[i])
                            
            
    away_i, home_i, i_away, i_home = [], [], [], []
    for i,j in enumerate(totals_away):
            if float(j) <= 3 and len(pitcher_away) < len(team_away):
                away_i.append(i)
            #elif float(j) >= 11 and len(pitcher_away) > len(team_away):
                #i_away.append(i)    
    for i,j in enumerate(totals_home):
            if float(j) <= 3 and len(pitcher_home) < len(team_home):
                home_i.append(i)
            #elif float(j) >= 11 and len(pitcher_home) > len(team_home):
                #i_home.append(i)
    
    # Remove extra SP
    j, p = 0, 0            
    away_pop, home_pop = [], []            
    for i in away_i:
        pitcher_away.insert(i, 'Unknown')
    for i in home_i:
        pitcher_home.insert(i, 'Unknown')
    for i in i_away:
        away_pop.append(pitcher_away.pop(i+1-j))
        j+=1
    for i in i_home:
        home_pop.append(pitcher_home.pop(i+1-p))
        p+=1
        
    if len(pitcher_away) > len(team_away) and 'SHOHEI (H) OHTANI' in pitcher_away:
        pitcher_away.remove('SHOHEI (H) OHTANI')
    if len(pitcher_home) > len(team_home) and 'SHOHEI (H) OHTANI' in pitcher_home:
        pitcher_home.remove('SHOHEI (H) OHTANI')

    temp, humidity, rain = [], [], []
    for cat in weather:
        if '°' in cat and 'H' in cat and cat.count('%') > 1:
            temp.append(cat.split()[0])
            humidity.append(cat.split()[1])
            rain.append(cat.split()[2])
        elif '°' in cat and 'H' in cat and cat.count('%') <= 1:
            temp.append(cat.split()[0])
            humidity.append(cat.split()[1])
            rain.append('0%')
        elif '°' in cat and 'H' not in cat and cat.count('%') <= 1:
            temp.append(cat.split()[0])
            humidity.append('H25%')
            rain.append(cat.split()[2])
        elif '°' not in cat and 'H' in cat and cat.count('%') > 1:
            temp.append('75°')
            humidity.append(cat.split()[1])
            rain.append(cat.split()[2])
        else:
            temp.append('75°')
            humidity.append('H25%')
            rain.append('0%')
    
    # Create Dataframes
    games = pd.DataFrame(np.column_stack([dates, team_away, team_home, pitcher_away, pitcher_home, expected_runs_away, actual_runs_away, expected_runs_home, actual_runs_home, temp, humidity, rain]), columns = ['date', 'team_away', 'team_home', 'pitcher_away', 'pitcher_home', 'expected_runs_away', 'actual_runs_away', 'expected_runs_home', 'actual_runs_home', 'temp', 'humidity', 'rain'])
        
        
    # Get game info - Lineups
    i=0
    player_dict, pos_dict, team_dict, starters_dict = {}, {}, {}, {}
    starting_player, team_ini, team_starters, team_pos, player = [], [], [], [], []
    for i,row in enumerate(soup.find_all('td')):
        if row.get_text() == 'Y':
            starting_player.append([soup.find_all('td')[i-4].get_text(), soup.find_all('td')[i-3].get_text(), soup.find_all('td')[i-1].get_text()])
            team_pos.append(soup.find_all('td')[i-3].get_text())
            team_ini.append(soup.find_all('td')[i-1].get_text())
            team_starters.append(soup.find_all('td')[i-4].get_text())
            #team_starters.append([starting_player, team_ini, team_pos])
            print(soup.find_all('td')[i+20].get_text(),  soup.find_all('td')[i+18].get_text(), soup.find_all('td')[i-16].get_text(), soup.find_all('td')[i-18].get_text(), soup.find_all('td')[i-19].get_text(), soup.find_all('td')[i-21].get_text(), soup.find_all('td')[i-22].get_text(), soup.find_all('td')[i+15].get_text(), soup.find_all('td')[i+14].get_text(), soup.find_all('td')[i+17].get_text())
            if int(soup.find_all('td')[i+20].get_text()) > 0 and int(soup.find_all('td')[i+20].get_text()) < 10 and soup.find_all('td')[i+18].get_text() != 'Y' and soup.find_all('td')[i-16].get_text() != 'Final':
                starting_player.append([soup.find_all('td')[i+14].get_text(), soup.find_all('td')[i+15].get_text(), soup.find_all('td')[i+17].get_text()])
                team_pos.append(soup.find_all('td')[i+15].get_text())
                team_ini.append(soup.find_all('td')[i+17].get_text())
                team_starters.append(soup.find_all('td')[i+14].get_text())
            if soup.find_all('td')[i-16].get_text() != 'Final' and soup.find_all('td')[i-16].get_text() != '\xa0' and int(soup.find_all('td')[i-16].get_text()) > 0 and int(soup.find_all('td')[i-16].get_text()) < 10 and soup.find_all('td')[i-18].get_text() != 'Y':
                starting_player.append([soup.find_all('td')[i-22].get_text(), soup.find_all('td')[i-21].get_text(), soup.find_all('td')[i-19].get_text()])
                team_pos.append(soup.find_all('td')[i-21].get_text())
                team_ini.append(soup.find_all('td')[i-19].get_text())
                team_starters.append(soup.find_all('td')[i-22].get_text())
    
    # Convert Team Initials
    starting_player = list(map((lambda x:[x[0],x[1],get_team_ini(x[2])]), starting_player))
    
    # Create Lineup DataFrame
    lineups = pd.DataFrame(np.column_stack([team_ini, team_starters, team_pos]), columns = ['team_ini', 'starters', 'team_pos'])

    # Place lineups into dictionarys
    names, pos, teams, starters_dict, player_list, player_dict, lineups_day = [],[],[],{},[],{},{}
    i,k=0,0
    tm = starting_player[0][2]
    for player,position,team in starting_player:
        if team == tm:
            names.append(player)
            pos.append(position)
            teams.append(team)
        else:
            for name,po in zip(names,pos):
                player_dict[name] = po
            player_list.append(player_dict)
            starters_dict[tm] = player_dict
            names,pos,teams,player_dict = [], [], [],{}
            names.append(player)
            pos.append(position)
            teams.append(team)
            tm = team
    for name,po in zip(names,pos):
        player_dict[name] = po
    player_list.append(player_dict)
    starters_dict[tm] = player_dict
    starters_dict = {k: v for k, v in starters_dict.items() if k is not None}        
    lineups_day[date] = starters_dict
    
    if scraped == 0:
        games_total = games
        all_lineups = {}
        all_lineups[date] = lineups_day[date]
        scraped = 1
    else:
        games_total = pd.concat([games_total, games], axis = 0)
        all_lineups[date] = lineups_day[date]

    return games_total, date, lineups_day, all_lineups

game_info = separate_games(soup)
game_data = game_info[0]
date = game_info[1]
daily_lineups = game_info[2]
lineups_all = game_info[3]

while date != '08-27-2018':
    elem1 = driver.find_element_by_name("BACK")
    elem1.click()
    time.sleep(20)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    game_info = separate_games(soup, game_data, lineups_all, 1)
    game_data = game_info[0]
    date = game_info[1]    
    daily_lineups = game_info[2]
    lineups_all = game_info[3]

        
game_data['team_away'] = list(map((lambda x:get_team_ini(x)), game_data['team_away']))
game_data['team_home'] = list(map((lambda x:get_team_ini(x)), game_data['team_home']))
game_data['temp'] = list(map((lambda x:x[:-1]), game_data['temp']))
game_data['humidity'] = list(map((lambda x:x[1:-1]), game_data['humidity']))
game_data['rain'] = list(map((lambda x:x[:-1]), game_data['rain']))
game_data['league'] = list(map((lambda x:get_team_league(x)), game_data['team_away']))
#game_data1['league'] = list(map((lambda x:get_team_league(x)), game_data1['team_away']))

lineups = pd.read_csv("MLB/lineups_all.csv", encoding='latin-1')
game_data1 = pd.read_csv("MLB/game_date_info1.csv", encoding='latin-1')
    
#game_data.to_csv('C:/Users/mrcrb/source/repos/MLB/game_date_info1.csv', sep=',', encoding='utf-8', index=False)
#game_data.to_csv('MLB/game_date_info1.csv', sep=',', encoding='utf-8', index=False)

#game_data1 = game_data1.iloc[:,:-1]

# Combine new and old dictionary
old = daily_lineups_pitch()
big_dict = {**lineups_all, **old}

with open('MLB/lineups_all.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, big_dict.keys(), extrasaction='ignore')
    w.writeheader()
    w.writerow(big_dict)

            
with open(r'MLB/game_date_info1.csv', 'a', newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    for i,row in game_data.iterrows():
        for j,row1 in game_data1.iterrows():
            if row[0] == row1[0] and row[1] == row1[1]:
                i = 0
                print(row[0], row[1])
                break
            else:
                i = 1
        if i == 1:
            writer.writerow(row)
            i = 0














with open('lineups_all.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, lineups_all.keys())
    for key in w:
        if key not in list(lineups_all.columns):
            w.writeheader()
            w.writerow(lineups_all)

with open('MLB/lineups_all1.csv','w') as f:
    w = csv.writer(f)
    w.writerows(lineups_all.items())

with open('lineups_all2.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, lineups_all)
    w.writeheader()
    w.writerows(lineups_all)

pd.DataFrame(lineups_all).to_csv('MLB/lineups_all3.csv', index=False)




s


# Get game info
i=0
player_dict, pos_dict, team_dict, starters_dict = {}, {}, {}, {}
starting_player, team_ini, team_starters, team_pos, player = [], [], [], [], []
for i,row in enumerate(soup.find_all('td')):
    if row.get_text() == 'Y':
        starting_player.append([soup.find_all('td')[i-4].get_text(), soup.find_all('td')[i-3].get_text(), soup.find_all('td')[i-1].get_text()])
        team_pos.append(soup.find_all('td')[i-3].get_text())
        team_ini.append(soup.find_all('td')[i-1].get_text())
        team_starters.append(soup.find_all('td')[i-4].get_text())
        #team_starters.append([starting_player, team_ini, team_pos])


starting_player = list(map((lambda x:[x[0],x[1],get_team_ini(x[2])]), starting_player))


names, pos, teams, starters_dict, player_list, player_dict = [],[],[],{},[],{}
i,k=0,0
tm = starting_player[0][2]
for player,position,team in starting_player:
    if team == tm:
        names.append(player)
        pos.append(position)
        teams.append(team)
        print(i, player,position,team)
    else:
        for name,po in zip(names,pos):
            player_dict[name] = po
        player_list.append(player_dict)
        starters_dict[tm] = player_dict
        names,pos,teams,player_dict = [], [], [],{}
        names.append(player)
        pos.append(position)
        teams.append(team)
        tm = team
for name,po in zip(names,pos):
    player_dict[name] = po
player_list.append(player_dict)
starters_dict[tm] = player_dict
        
lineups_day = {date:starters_dict}

# Create Dataframes
lineups = pd.DataFrame(np.column_stack([team_ini, team_starters, team_pos]), columns = ['team_ini', 'starters', 'team_pos'])

            
#player_list,player_dict = [],{}        
        
        list([names, pos])
    starters_dict[team] = {name: {'position' : pos}}).append(starters_dict[team])
        player_dict[soup.find_all('td')[i-4].get_text()] = {'position' : soup.find_all('td')[i-3].get_text(), 'team' : soup.find_all('td')[i-1].get_text()}
        if len(team_dict[soup.find_all('td')[i-1].get_text()].items()) > 0:
            team_dict[soup.find_all('td')[i-1].get_text()] = list({soup.find_all('td')[i-4].get_text() : player_dict[soup.find_all('td')[i-4].get_text()]}).append(team_dict[soup.find_all('td')[i-1].get_text()])
        else:
            team_dict[soup.find_all('td')[i-1].get_text()] = player_dict[soup.find_all('td')[i-4].get_text()]

        #team_dict[team_ini.append(soup.find_all('td')[i-1].get_text())]['batter'] = starting_player.append(soup.find_all('td')[i-4].get_text())
        #team_dict[team_ini.append(soup.find_all('td')[i-1].get_text())]['position'] = team_pos.append(soup.find_all('td')[i-3].get_text())
        #starting_player = []

pos_dict['position'] = team_pos.append(soup.find_all('td')[i-3].get_text())
player_dict['batter'] = starting_player.append(soup.find_all('td')[i-4].get_text())

'''
temp, humidity, rain = [], [], []
for cat in weather:
    if '°' in cat and 'H' in cat and cat.count('%') > 1:
        temp.append(cat.split()[0])
        humidity.append(cat.split()[1])
        rain.append(cat.split()[2])
    elif '°' in cat and 'H' in cat and cat.count('%') <= 1:
        temp.append(cat.split()[0])
        humidity.append(cat.split()[1])
        rain.append('0%')
    elif '°' in cat and 'H' not in cat and cat.count('%') <= 1:
        temp.append(cat.split()[0])
        humidity.append('25%')
        rain.append(cat.split()[2])
    elif '°' not in cat and 'H' in cat and cat.count('%') > 1:
        temp.append('75°')
        humidity.append(cat.split()[1])
        rain.append(cat.split()[2])
    else:
        temp.append('75°')
        humidity.append('25%')
        rain.append('0%')
'''        


driver.close()
driver.quit()