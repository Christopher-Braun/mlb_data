#! python3
from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
import re


# Importing the dataset
starting_lineups = pd.read_csv("MLB/starting_lineups.csv", encoding='latin-1')
starting_lineups.info()

teams = []
for team in starting_lineups['Tm']:
    teams.append(team[:3])

starting_lineups = starting_lineups.iloc[:,1:]

seperate, group = [], []
for name in starting_lineups.items():
    for char in name:
        for c in char:
           d = c.split(" ")
           seperate.append(d[0])



