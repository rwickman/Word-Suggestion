from bs4 import BeautifulSoup
import requests, re, json

""" Script I used to scrap all the movie quotes from this website to use a for data"""

r = requests.get("https://www.infoplease.com/arts-entertainment/movies-and-videos/top-100-movie-quotes")

soup = BeautifulSoup(r.text, "html.parser")
ol = soup.find_all('ol')[1]

movie_quotes = []

for li in ol.find_all("li"):
    # quote = li.p.b.text
    # if quote:
    #     print(quote)
    line = li.p.text
    quote = ""
    quote_match = re.search("(?<=“).+(?=”)", line)
    if quote_match:
        quote = quote_match.group(0)
    else:
        continue
    movie = re.search("(?<=” ).+(?=, \d)", line).group(0)
    if quote:
        movie_quotes.append(
            {
                "movie:" : movie,
                "quote" : quote
            }
        )

with open("../../data/movie_quotes.json", 'w') as f:
    json.dump(movie_quotes, f)
