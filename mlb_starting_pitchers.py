#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd

"""Individual Pitching"""
wiki = "http://www.espn.com/mlb/schedule"
page = urllib.request.urlopen(wiki)
soup = BeautifulSoup(page, 'lxml')

print(soup.prettify())

wins, text = [], []
for row in soup.select('a[name="&lpos=mlb:schedule:player"]'):
    wins.append(row.get_text())
    text.append(row.get('href'))

ids, name = [], []
for t in text:   
    z=t.split("/")
    ids.append(z[7])
    name.append(z[8])


dates = []
for row in soup.select('h2[class="table-caption"]'):
    dates.append(row.get_text())
    
pit_win = []
for row in soup.select('td[class="pitcher_win"]'):
    pit_win.append(row.a)



weather = []
for row in soup.find_all("tbody"):
     weather.append(row.get_text())
     
odd = []
for row in soup.select('tr'):
    odd.append(row.get_text())
    

mid = []    
for i,o in enumerate(odd):
    if o == 'matchupresultwinlosssave':
        start = i+1
    elif o == 'matchuptime\xa0nat tvpitching matchuptickets':
        mid.append(i+1)

if 'matchupresultwinlosssave' in odd: 
    odd.remove('matchupresultwinlosssave')
if 'matchuptime\xa0nat tvpitching matchuptickets' in odd:
    odd.remove('matchuptime\xa0nat tvpitching matchuptickets')
if 'matchuptime\xa0nat tvpitching matchuptickets' in odd:
    odd.remove('matchuptime\xa0nat tvpitching matchuptickets')

odd=odd[:16]


date = []
for i in range(len(odd)):
    if i < mid[0]:
        date.append(dates[0])
    elif i < mid[1] and i >= mid[0]:
        date.append(dates[1])
    elif i >= mid[1]:
        date.append(dates[2])    
            
even, odds = [], []
for i in range(len(ids)):
    if i % 2 == 0:
        even.append(i)
    else:
        odds.append(i)

win, lose = [], []
for i,j in zip(even, odds):
    win.append(ids[i])
    lose.append(ids[j])
     
ps = []    
for o in odd:
    p = o.split(',')
    if len(p)>1:
        ps.append(p[1])

psp = []
for word in ps:
    for i in range(len(word)):
        if word[i].isnumeric():
            wordi = word[i]
            words = word.split(wordi)
            if len(words[1])>0 and words[1][0].isupper():
                psp.append(words[1])
            else:
                word2 = words[1].split(")")
                if len(word2)>1:
                    psp.append(word2[1])
            
     
cword = []
for c in psp:
    i=1
    for i in range(1,len(c)):
        lc = len(c)
        if c[i-1].islower() and c[i].isupper():
            print(i)
            cs = c[:i]
            c = c[:i] + ' ' + c[i:]
            #csss = cs + ' ' + c
    cword.append(c)
    
xt = str()    
cname = []    
for c in cword:
    x = c.split(' ')
    for i in range(len(x)):
        xtt = x[i]
        xt = xt.lstrip() + ' ' + xtt
    cname.append(xt)
    xt = str()
    
#cname[4] = 'Danny Barnes Zach McAllister'

d1, d2 = [], []
for row in cname:
    d = row.split(' ')
    d1.append(d[0] + ' ' + d[1])
    d2.append(d[2] + ' ' + d[3])


win_pitch = pd.DataFrame(np.column_stack([d1, d2]), columns = ['WINNER', 'LOSER'])
win_pitch.to_csv('MLB/win_pitch.csv', sep='\t', encoding='utf-8')

            
            
    






