import pandas as pd
import numpy as np

X = pd.read_csv("game_bat_data.csv", encoding='latin-1')
game_df = pd.read_csv("game_data.csv", encoding='latin-1')


class Lineup(object):
    """Basic Player Information - Name, Team, Position, Hand...
    """
    #players_df = pd.read_csv("MLB/batter_all.csv", encoding='latin-1')
    
    def __init__(self, data, team):
        self.team = team
        self.data = data
        self.players = [x[1] for x in self.data[self.data['Team'] == self.team].iterrows()]
        #self.name = name
        #self.position = position
        #[x for x in self.players]
        
    def __iter__(self):
        """Allows object to be iterated over."""
        for x in self.players:
            yield x
            
    def print_players(self):
        for x in self.players:
            return x
            
    def return_players(self):
        for x in self.players:
            self.name = x.Name
            self.position = x.Pos
            return self.name, self.position, self.team
        
    def __str__(self):
        return self.players
        

class Player(object):
    """Basic Player Information - Name, Team, Position, Hand...
    """
    #players_df = pd.read_csv("MLB/batter_all.csv", encoding='latin-1')
    
    def __init__(self, data):
        self.player = data.Name
        self.position = data.Pos
        self.team = data.Team
        
    def return_player(self):
        return {'player' : self.player}, {'position' : self.position}, {'team' : self.team}
    
    def __str__(self):
        return [self.player, self.position, self.team]
    
    
class Game(object):
    
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
        
        pass

from collections import defaultdict
teams = defaultdict(list)
for i in X.Team.unique():
    [teams[i].append(Player(p).return_player()) for p in Lineup(X,i)]

teams_list = defaultdict(list)
for i in X.Team.unique():
    [teams_list[i].append(Player(p)) for p in Lineup(X,i)]

games1 = defaultdict(list)
for game in game_df.iterrows():
    games1[game[1].game_id].append(Game(game[1]))