from bs4 import BeautifulSoup
import urllib.request
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd


web = "https://legacy.baseballprospectus.com/standings/index.php?odate=2018-04-10&otype=csv"
adj_standings = urllib.request.urlopen(web)   
standings = pd.read_csv(adj_standings, encoding='latin-1')
standings.to_csv('MLB/standings.csv', sep='\t', encoding='utf-8')

