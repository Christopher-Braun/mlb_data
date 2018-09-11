from selenium import webdriver
driver = webdriver.Chrome()
driver.get("https://baseballmonster.com/boxscores.aspx")
elem1 = driver.find_element_by_name("BACK")
elem1.click()
html = driver.page_source
