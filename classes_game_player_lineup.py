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
        self.home_lineup = []
        self.away_lineup = []
        
        pass

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

def add_lineups(data):
    # Add home and away lineups
    for key in data.keys():
        for game in data[key]:
            game.home_lineup = teams_list[game.home_team]
            game.away_lineup = teams_list[game.away_team]
    return data


teams = daily_lineup_dict(X)
teams_list = daily_lineup(X)
games = games_dict(game_df)
games = add_lineups(games)
        