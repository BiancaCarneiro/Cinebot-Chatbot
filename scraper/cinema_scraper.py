import requests
from bs4 import BeautifulSoup
from ratelimit import limits, sleep_and_retry

MAX_CALLS_PER_SECOND = 1
PERIOD = 1
CINEMA_URL = "https://www.cinemark.com.br/recife/cinemas"

class CinemaSpider:

    def __init__(self) -> None:
        self.url:str = CINEMA_URL
        self.movies_dir:dict = dict()
        self.movies_prices:str = str()

    @sleep_and_retry
    @limits(calls=MAX_CALLS_PER_SECOND, period=PERIOD)
    def _limiter(self):
        return

    def _scrap_movies_info(self) -> None:
        self._limiter()
        
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, 'html.parser')

        movies_dir = {}
        movie_blocks = soup.find_all('div', class_= 'theater')

        for movie_block in movie_blocks:
            movies_dir[movie_block.find("h3").get_text(strip=True)] = [time.get_text(strip=True) for time in movie_block.find_all('ul', class_='times-options')]

        self.movies_dir = movies_dir

    def _scrap_price_info(self):
        self._limiter()
        
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, 'html.parser')

        prices = soup.find('div', id='theater-prices-2107')
        self.movies_prices = prices.get_text()
        
    def get_prices(self) -> str:
        if not self.movies_prices:
            self._scrap_price_info()
        return self.movies_prices
    
    def get_times_dir(self) -> dict:
        if not self.movies_dir:
            self._scrap_movies_info()
        return self.movies_dir
        


def main():
    spider = CinemaSpider()
    spider.get_movies_info()
    print(spider.movies_dir)
    spider.get_price_info()
    print(spider.movies_prices)


if __name__ == "__main__":
    main()
