from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd


"""The home page for MLB"""
"""Adjusted Standings"""
wiki = "http://legacy.baseballprospectus.com/standings/index.php?dispgroup=all&submit=Go"
page = urllib.request.urlopen(wiki)
soup = BeautifulSoup(page, 'lxml')

teams, name = [], []
for row in soup.find_all('option'):
	if row not in teams:
		teams.append(row.get('value'))
		team_name = row.get('value')
		name.append(row.get_text())
		full_name = row.get_text()
		
i = 0
percent, wins, losses, actual, first, second, third, win, wins1, wins2, wins3, loss, loss1, loss2, loss3 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for row in soup.find_all('center'):
    if row.get_text()[0:1] == '.':
        percent.append(row.get_text()[0:4])
        for i in range(len(row.get_text()[4:].split('-'))):
            if i == 0:
                wins.append(row.get_text()[4:].split('-')[i])
            else:
                losses.append(row.get_text()[4:].split('-')[i])

actual = [percent[i] for i in range(0,120,4)]
first = [percent[i] for i in range(1,120,4)]
second = [percent[i] for i in range(2,120,4)]   
third = [percent[i] for i in range(3,120,4)]
win = [wins[i] for i in range(0,120,4)]
win1 = [wins[i] for i in range(1,120,4)]
win2 = [wins[i] for i in range(2,120,4)]   
win3 = [wins[i] for i in range(3,120,4)]
loss = [losses[i] for i in range(0,120,4)]
loss1 = [losses[i] for i in range(1,120,4)]
loss2 = [losses[i] for i in range(2,120,4)]   
loss3 = [losses[i] for i in range(3,120,4)]

             
standings = []
for row in soup.find_all('b'):
    if row.get_text() in name:
        standings.append(row.get_text())

stand_team = []
for i in standings:
    for j in name:
        if i == j:
            stand_team.append(teams[name.index(i)])
            
        
league = pd.DataFrame(np.column_stack([standings, stand_team, actual, first, second, third, win, win1, win2, win3, loss, loss1, loss2, loss3]), columns = ['Team', 'ABV', 'Win %', '1st Order Win %', '2nd Order Win %', '3rd Order Win %', 'Wins', 'Wins1', 'Wins2', 'Wins3', 'Losses', 'Losses1', 'Losses2', 'Losses3'])        
              










      
from bs4 import SoupStrainer        

        try:
			team = teams.objects.get(name = team_name)
		except Team.DoesNotExist:
			team = None
			if team is None:
				team = Team(name=team_name, full_name=full_name)
				team.save()


	context = {'teams': teams, 'name': name, 'team': team}

	return render(request, 'mlb/index.html', context)



def players(request):

	wiki1 = "http://legacy.baseballprospectus.com/sortable/index.php?cid=1918873"
	page1 = urllib.request.urlopen(wiki1)
	soup1 = BeautifulSoup(page1, 'lxml')

	players = []
	for element in soup1.find_all('tr')[6:1330]:
		players.append(element.a.get_text())
		try:
			player = Player.objects.get(name = element.a.get_text())
			player.league = element.contents[3].get_text()
			player.games = element.contents[6].get_text()
			player.Tav = element.contents[34].get_text()
			player.VORP = element.contents[35].get_text()
			player.FRAA = element.contents[36].get_text()
			player.BWARP = element.contents[37].get_text()
			player.save()
		except Player.DoesNotExist:
			player = None
			if player is None:
				player = Player(name = element.a.get_text(),
				league = element.contents[3].get_text(),
				games = element.contents[6].get_text(),
				Tav = element.contents[34].get_text(),
				VORP = element.contents[35].get_text(),
				FRAA = element.contents[36].get_text(),
				BWARP = element.contents[37].get_text())
				tm = Team.objects.get(name = element.contents[2].get_text().lower())
				player.team = tm
				player.save()

	player_new = Player.objects.all()

	context = {'players': players, 'player_new': player_new}

	return render(request, 'mlb/players.html', context)








