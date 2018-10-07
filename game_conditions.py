import os
import time
from selenium import webdriver
import pandas as pd
import re
from classes_game_player_lineup import Game, Player, Lineup, daily_lineup_dict, daily_lineup, games_dict, add_lineups, separate_double_headers
from collections import defaultdict


def get_row_data(table):
    for row in table.find_elements_by_xpath(".//tr"):
        yield [td.text for td in row.find_elements_by_xpath(".//td")]

def date_match(date):
    patgrep = '(\d*)[/.-](\d{2})[/.-](\d{4})$'
    patc = re.compile(patgrep)
    current_date = patc.search(date).group().replace('/', '-')
    if current_date[1] == "-":
        return current_date.zfill(10)
    return current_date

def is_dome(d):
    d['Temp'], d['Humidity'], d['Rain'] = int(72), int(50), int(0)
    return d

def fix_weather(data, re_temp, re_humidity, re_rain):
    for d in data:
        if not d['Weather']:
            d = is_dome(d)
        else:
            d['Temp'], d['Humidity'] = int(re.search(re_temp, d['Weather']).group()), int(re.search(re_humidity, d['Weather']).group()[1:-1])
            p1 = re.search(re_rain, d['Weather']).group().strip().replace('%','')
            if not p1:
                d['Rain'] = int(0)
            else:
                d['Rain'] = int(p1)
        del d['Weather']
    return data

def fix_home(data):
    for d in data:
        d['Home'] = d['Home'].replace('@ ', '')
    return data

def create_game_id(data, date):
    for d in data:
        d['game_id'] = str(date) + str(d['Home'])
    return data

def add_game_date(data, date):
    for d in data:
        d['Date'] = str(date)
    return data

def current_date(current_date):
    if current_date is None:
        current_date = 1
    else:
        current_date = current_date
    return current_date

def main():
    # Set initial date
    current_date = 1
    games_total = defaultdict(list)
    
    # Selenium Initialize
    os.environ['MOZ_HEADLESS'] = '1'
    driver = webdriver.Firefox()
    driver.get("https://baseballmonster.com/boxscores.aspx")
    driver.find_element_by_name("BACK").click()
    time.sleep(2)
        
    # Start Scraping
    while current_date != '09-27-2018':
    
        # Header Data
        game_header_xpath = "//form[@id='form1']/div[5]/table[1]/tbody/tr[2]/td[2]/table/tbody/tr/td/table/tbody/tr/td/table[1]/thead/tr/th"
        bat_header_xpath = "//form[@id='form1']/div[5]/table[2]/tbody/tr[3]/td[1]/table/thead/tr/th"
        pitch_header_xpath = "//form[@id='form1']/div[5]/table[2]/tbody/tr[3]/td[2]/table/thead/tr/th"
        gheader_list = [header.text for header in driver.find_elements_by_xpath(game_header_xpath)]
        bheader_list = [header.text for header in driver.find_elements_by_xpath(bat_header_xpath)]
        pheader_list = [header.text for header in driver.find_elements_by_xpath(pitch_header_xpath)]
        print("Headers Done")
        
       # Game Data
        table1_id_xpath = "//table[@class='table table-bordered table-hover table-sm base-td-small datatable ml-0']"
        table_id_xpath = "//table[@class='table table-bordered table-hover table-sm base-td-small datatable ml-0 nowraptable ']"
        game_list = []
        batter_list = []
        pitcher_list = []
    
        # Date Data
        date = "//form[@id='form1']/div[4]/h1"
        date_list = [header.text for header in driver.find_elements_by_xpath(date)]
        current_date = date_match(date_list[0])
        print(current_date)
    
        for table in driver.find_elements_by_xpath(table1_id_xpath):
            print(f"Games Data:")
            game_list = game_list + [data for data in get_row_data(table) if len(data) == 10]
            print(len(game_list))  
    
        for table in driver.find_elements_by_xpath(table_id_xpath):
            print(f"Game Data:")
            batter_list = batter_list + [data for data in get_row_data(table) if len(data) == 18]
            pitcher_list = pitcher_list + [data for data in get_row_data(table) if len(data) == 19]
            print(len(batter_list))
            print(len(pitcher_list))
            
    
        # Attribute gheader_list runs and projected runs to home or away
        gheader_list[2], gheader_list[3], gheader_list[5], gheader_list[6] = 'pR_Away', 'aR_Away', 'pR_Home', 'aR_Home'
    
        # Create list of dictionaries for each player/row and remove "Totals" line
        game_data = [dict(zip(gheader_list[1:], l[1:])) for l in game_list]
        game_bat_data = [dict(zip(bheader_list, l)) for l in batter_list if not "Totals" in l]
        game_pitch_data = [dict(zip(pheader_list, l)) for l in pitcher_list if not "Totals" in l]
    
        # Remove blank rows
        game_bat_data = [d for d in game_bat_data if d["Name"] is not " "]
        game_pitch_data = [d for d in game_pitch_data if d["Name"] is not " "]
        
        # Add date
        game_bat_data = add_game_date(game_bat_data, current_date)
        game_pitch_data = add_game_date(game_pitch_data, current_date)
        
        # Split & Format Weather Data, Format Home
        m, n, p = '[0-9]{1,3}(?=°)', 'H[0-9]{1,3}%', '\s+[0-9]{1,3}%|(?<=H[0-9][0-9]%)'
        game_data = fix_weather(game_data, m, n, p)
        game_data = fix_home(game_data)
        
        # Create Game Id
        game_data = create_game_id(game_data, current_date)
        #game_pitch_data = create_game_id(game_pitch_data, current_date)
        
        # Create Dataframe
        bat_df = pd.DataFrame(game_bat_data)
        pitch_df = pd.DataFrame(game_pitch_data)
        game_df = pd.DataFrame(game_data)
        
        # Dataframes to classes
        teams_list = daily_lineup(bat_df)
        teams_list = separate_double_headers(teams_list)
        games = games_dict(game_df)
        
        # Add lineups to Game classes (struggling with double headers)
        # Double header games have a large teams_list (2 teams)
        # My solution was to split into 2 teams at the point a player in game 1 comes up a 2nd time in teams_list
        # Kind of a weak soln
        games = add_lineups(games, teams_list)
        games = add_starting_pitcher(games, pitch_df)
        

        # Add Dataframes
        for game in games:
            games_total[game] = games[game]

        # Save Dataframes
        bat_df.to_csv('game_bat_data.csv', index=False)
        pitch_df.to_csv('game_pitch_data.csv', index=False)
        game_df.to_csv('game_data.csv', index=False)

        driver.find_element_by_name("BACK").click()
        time.sleep(2)
    
    driver.quit()

if __name__ == '__main__':
        main()
        
        