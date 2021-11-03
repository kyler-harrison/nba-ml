from bs4 import BeautifulSoup as bs
import requests
import time


# desc.: generate urls for ALL years for each month on the page
# input: starting year, ending year
# output: list of urls to scrape
def gen_urls(start_yr, end_yr):
    # generate url list in order by month for one season (oct-jun)
    url_months = ["", "-november", "-december", "-january", "-february", "-march", "-april", "-may", "-june"]
    url_years = [str(year) for year in range(start_yr, end_yr)]

    all_urls = []  # list of lists (each sublist is a season of urls)
    for year in url_years:
        season_urls = []  # urls for one season
        for month in url_months:
            season_urls.append("https://www.basketball-reference.com/leagues/NBA_" + year + "_games" + month + ".html")
        all_urls.append(season_urls)

    return all_urls


def get_data(url):
    # get page, the agent thing allows to bypass permission denied error
    agent = {"User-Agent": 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
    data = requests.get(url, headers=agent)
    # load data into bs
    soup = bs(data.text, "html.parser")

    month_games = []
    for tr in soup.find_all("tr"):
        game_ls = [td.text for td in tr.find_all("td")]
        if len(game_ls) == 9:
            month_games.append(game_ls)

    return month_games


# input: list of months of games
def write_games(all_games, file_path, year):
    file = open(file_path, "w")
    header = "year,team_0,team_1,team_0_score,team_1_score,score_diff,outcome\n"
    file.write(header)
    for month in all_games:
        for game in month:
            # outcome is -1 if team_0 won, 1 if team_1 won
            difference = float(game[2]) - float(game[4])
            tie = False
            if difference < 0:
                outcome = -1
            elif difference > 0:
                outcome = 1
            else:
                tie = True
            if not tie:
                line = year + "," + game[1] + "," + game[3] + "," + game[2] + "," + game[4] + "," + str(difference) + "," + str(outcome) + "\n"
                file.write(line)

    file.close()


def main():
    total_start = time.time()
    urls = gen_urls(2001, 2020)  # get urls from 2001-2019
    # get file name for each season
    file_names = ["data/" + str(year) + "_games.csv" for year in range(2001, 2020)]

    for i, url_list in enumerate(urls):
        season_start = time.time()
        all_games = []
        for url in url_list:
            all_games.append(get_data(url))
        write_games(all_games, file_names[i], file_names[i][5:9])

        season_end = time.time()
        print("Season: {}\nTime (sec): {}\n".format(file_names[i][5:9], season_end - season_start))

    total_end = time.time()
    print("Total time (sec): {}".format(total_end - total_start))


if __name__ == "__main__":
    main()
