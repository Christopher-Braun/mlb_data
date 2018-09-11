from requests_html import HTMLSession
session = HTMLSession()
from bs4 import BeautifulSoup
import urllib.request


r = session.get('https://www.baseball-reference.com/leagues/MLB/2018-schedule.shtml')

r.html.links

r.html.absolute_links

links = []
for html in r.html:
    links.append(html.clean)

boxes1 = r.html.find('a', containing = "box", clean = True)

href = list(map((lambda x:x.split("'/")[1]),boxes))

boxes_str = str(boxes1)

link  = []
for box in boxes1:
    link.append(str(box))
    
box = list(map((lambda x:x.split("'/")[1]),link))
html = list(map((lambda x:x.split("'>")[0]),box))
total = list(map((lambda x:'https://www.baseball-reference.com/' + x),html))

boxes = r.html.xpath('//a', first=True)


n = session.get('https://www.baseball-reference.com/boxes/TOR/TOR201804250.shtml')

weather = n.html.xpath('//*[@id="div_9100295389"]/div[4]/text()')

n.html.render()
n.html.search('Start Time Weather:')
wiki='https://www.baseball-reference.com/boxes/TOR/TOR201804250.shtml'
page = urllib.request.urlopen(wiki)
soup = BeautifulSoup(page, 'lxml')
so = soup.strong
s1=so.find_all_next(string=True)
tried = n.html
x1= soup.find_all('strong')




