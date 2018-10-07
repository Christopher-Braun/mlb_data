import pandas as pd
import numpy as np

#X = pd.read_csv("game_bat_data.csv", encoding='latin-1')
#game_df = pd.read_csv("game_data.csv", encoding='latin-1')
#players_df = pd.read_csv("MLB/batter_all.csv", encoding='latin-1')
#players_df = pd.read_csv("MLB/batter_all.csv", encoding='latin-1')
#players_df = pd.read_csv("MLB/batter_all.csv", encoding='latin-1')

class Lineup(object):
    """Basic Lineup Information - Players, Team, Position
    """
    
    def __init__(self, data, team):
        self.team = team
        self.data = data
        self.players = [x[1] for x in self.data[self.data['Team'] == self.team].iterrows()]
        #self.position = position
        
    def __iter__(self):
        """Allows object to be iterated over."""
        for x in self.players:
            yield x
    '''         
    def print_players(self):
        for x in self.players:
            return x
           
    def return_players(self):
        for x in self.players:
            self.name = x.Name
            self.position = x.Pos
            return self.name, self.position, self.team
    '''    
    def __str__(self):
        return self.players
        

class Player(object):
    """Basic Player Information - Name, Team, Position, Hand...
    """
    
    def __init__(self, data):
        self.player = data.Name
        self.position = data.Pos
        self.team = data.Team
        self.avg = float()
        self.hits = int()
        self.homeruns = int()
        self.runs = int()
        
    def return_player(self):
        '''Return dictionary of player attributes'''
        return {'player' : self.player}, {'position' : self.position}, {'team' : self.team}
    
    def __str__(self):
        return [self.player, self.position, self.team]

class Pitcher(object):
    """Basic Player Information - Name, Team, Position, Hand...
    """
    
    def __init__(self, data):
        self.game_id = data.game_id
        self.player = data.Name
        self.position = data.Pos
        self.team = data.Team
        self.value = data.Value
        self.era = float()
        self.hits = int()
        self.homeruns = int()
        self.runs = int()
        
    def __iter__(self):
        """Allows object to be iterated over."""
        for x in self.data:
            yield x        
        
    def return_player(self):
        '''Return dictionary of pitcher attributes'''
        return {'player' : self.player}, {'position' : self.position}, {'team' : self.team}
    
    def __str__(self):
        return [self.player, self.position, self.team]    
    
class Game(object):
    '''Game Information - game id, home team, away team, temp...'''
    def __init__(self, data):
        self.game_id = data.game_id
        self.home_team = data.Home
        self.away_team = data.Away
        self.time = data.Time
        self.temp = data.Temp
        self.humidity = data.Humidity
        self.rain = data.Rain
        self.home_runs = data.aR_Home
        self.home_predict_runs = data.pR_Home
        self.away_runs = data.aR_Away
        self.away_predict_runs = data.pR_Away
        self.game_status = data.Status
        self.home_lineup = []
        self.away_lineup = []
        self.starting_pitcher = str()
        self.starting_pitcher_instance = object()
        
        def __iter__(self):
            """Allows object to be iterated over."""
            for x in self.data:
                yield x
        '''
        def __iter__(self):
            return self
        '''
        def add_starting_pitcher(self, pitchers):
            '''attempting to create a class method that adds starting_pitcher_instance - DOESN'T WORK'''
            self.starting_pitcher_instance = Pitcher(pitchers[pitchers['Name']==self.starting_pitcher][pitchers['game_id']==self.game_id])     
        
        def __str__(self):
            return [self.game_id, 
                    self.home_team, 
                    self.away_team, 
                    self.time,
                    self.temp,
                    self.humidity,
                    self.rain,
                    self.home_runs,
                    self.home_predict_runs,
                    self.away_runs,
                    self.away_predict_runs,
                    self.game_status,
                    self.home_lineup,
                    self.away_lineup,
                    self.starting_pitcher]
            

from collections import defaultdict

def daily_lineup_dict(data):
    # Create lineup dictionary for each team on day of most recent scraping
    teams = defaultdict(list)
    for i in data.Team.unique():
        [teams[i].append(Player(p).return_player()) for p in Lineup(data,i)]
    return teams

def daily_lineup(data):
    # Create lineup list for each team on day of most recent scraping
    teams_list = defaultdict(list)
    for i in data.Team.unique():
        [teams_list[i].append(Player(p)) for p in Lineup(data,i)]
    return teams_list

def games_dict(data):
    # Create dictionary of games with info from game class
    games = defaultdict(list)
    for game in data.iterrows():
        games[game[1].game_id].append(Game(game[1]))
    return games

def pitchers_dict(data):
    # Create dictionary of games with info from game class
    pitchers = defaultdict(list)
    for pitcher in data.iterrows():
        pitchers[pitcher[1].game_id].append(Pitcher(pitcher[1]))
    return pitchers

def add_lineups(data, teams_list):
    # Add home and away lineups to game info
    for key in data.keys():
        i = 1
        for game in data[key]:
            if len(data[key])<2:
                game.home_lineup = teams_list[game.home_team]
                game.away_lineup = teams_list[game.away_team]
            else:
                if i == 1:
                    game.home_lineup = teams_list[game.home_team][0]
                    game.away_lineup = teams_list[game.away_team][0]
                    i+=1
                else:
                    game.home_lineup = teams_list[game.home_team][1]
                    game.away_lineup = teams_list[game.away_team][1]
    return data

def separate_double_headers(data):
    # Split the lineup based on the 1st repeated player and insert into team_lineups
    for team in data:
        players = []
        for player in data[team]:
            if player.player in players:
                data[team] = [data[team][:len(players)], data[team][len(players):]]
                players = []
                break
            else:
                players.append(player.player)
    return data

def add_starting_pitcher(data, pitchers):
    # Add starting pitcher's name to game
    for key in data.keys():
        i = 1
        for game in data[key]:
            if len(data[key])<2:
                game.starting_pitcher = pitchers['Name'][pitchers['Pos']=='SP'][pitchers['Team']==game.home_team].iloc[0]
            else:
                if i == 1:
                    game.starting_pitcher = pitchers['Name'][pitchers['Pos']=='SP'][pitchers['Team']==game.home_team].iloc[0]
                    print(game.starting_pitcher)
                    i+=1
                else:
                    game.starting_pitcher = pitchers['Name'][pitchers['Pos']=='SP'][pitchers['Team']==game.home_team].iloc[1]
    return data

def add_starting_pitcher2(data, pitchers):
    # Add starting pitcher info to game
    for key in data.keys():
        for game in data[key]:
            game.starting_pitcher_instance = Pitcher(pitchers[pitchers['Name']==game.starting_pitcher][pitchers['game_id']==game.game_id].iloc[0])
    return data



'''
Pitcher(pitch_df[pitch_df['Pos']=='SP'][pitch_df['game_id']==self.game_id])

p1 = Game(game_df.iloc[0:1])
p2 = Game(game_df.iloc[1:2])
test = pd.Series(game_df['Away'].iloc[3:4])

l = len(players)
team1 = data[team][:len(players)]
team2 = data[team][len(players):]
data[team] = [team1, team2]
print(team1[0].player, team2[0].player, l, team1[9].player, team2[9].player)

game_list = []
batter_list = []
pitcher_list = []
game_list, batter_list, pitcher_list = [], [], []
data[team] = [data[team][:len(players)], data[team][len(players):]]

# Add lineups to Game classes (struggling with double headers)
# Double header games have a large teams_list (2 teams)
# My solution was to split into 2 teams at the point a player in game 1 comes up a 2nd time in teams_list
# Kind of a weak soln

def add_starting_pitcher1(data, pitchers):
    # Add home and away lineups
    for key in data.keys():
        i = 1
        for game in data[key]:
            if len(data[key])<2:
                #p1 = Pitcher()
                #p1.player = pitchers['Name'][pitchers['Pos']=='SP'][pitchers['Team']==game.home_team].iloc[0]
                #game.starting_pitcher = pitchers['Name'][pitchers['Pos']=='SP'][pitchers['Team']==game.home_team].iloc[0]
                game.starting_pitcher = [pitch.player for pitch in pitchers[key] if pitch.position == 'SP']
            else:
                if i == 1:
                    game.starting_pitcher = [pitch.player for pitch in pitchers[key] if pitch.position == 'SP'][0]
                    #game.starting_pitcher = pitchers['Name'][pitchers['Pos']=='SP'][pitchers['Team']==game.home_team].iloc[0]
                    print(game.starting_pitcher)
                    i+=1
                else:
                    game.starting_pitcher = [pitch.player for pitch in pitchers[key] if pitch.position == 'SP'][1]
                    #game.starting_pitcher = pitchers['Name'][pitchers['Pos']=='SP'][pitchers['Team']==game.home_team].iloc[1]
    return data


# Trying to figure out why I can't iterate
for game in games:
    if len(games[game])<2:
        for day in games[game]:
            print('Nope')
            #game.home_lineup = teams_list[game.home_team]
            #game.away_lineup = teams_list[game.away_team]
    else:
        for day in games[game]:
            print(day.away_lineup)
            if game1 == game_id:
                game1 = day.game_id
            print(day.__dict__)
            print(game.index())
            print(game.exec())
            
            
            #print(game.__subclasses__())
            #game.home_lineup = teams_list[game.home_team][0]
            #game.away_lineup = teams_list[game.away_team][0]
        for game in games[key][1]:
            print(game)
            #game.home_lineup = teams_list[game.home_team][1]
            #game.away_lineup = teams_list[game.away_team][1]
            #game[0].home_lineup = teams_list[game[0].home_team][0]
            #game[0].away_lineup = teams_list[game[0].away_team][0]
            #game[1].home_lineup = teams_list[game[1].home_team][1]
            #game[1].away_lineup = teams_list[game[1].away_team][1]

    
for game in games:
    print(games[game][0])   
    
'''
'''
teams = daily_lineup_dict(X)
teams_list = daily_lineup(X)
games = games_dict(game_df)
games = add_lineups(games, teams_list)
'''        