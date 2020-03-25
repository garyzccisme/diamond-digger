import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import platform
import os
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

class Scrapper:
    def __init__(self, url):
        self.url = url

    def get(self):
        return


class RequestsScrapper(Scrapper):
    def __init__(self, url=None, headers=None):
        super().__init__(url)
        self.url = url
        self.headers = headers
        self.raw = None
        self.df = None

    def get(self):
        s = requests.session()
        self.raw = BeautifulSoup(s.get(self.url, headers=self.headers).content, "html.parser")

        headerinctags = self.raw.find_all('div', class_='grid-header normal-header')
        header = headerinctags[0].get_text(';').split(";")[1:]
        diamondsmessy = self.raw.find_all('a', class_='grid-row row')

        # Start to construct the DataFrame
        diamonds = pd.DataFrame(columns=header)
        for i in range(len(diamondsmessy)):
            a = diamondsmessy[i].get_text(";")
            b = a.split(";")
            del b[4]
            a = pd.DataFrame(b, index=header)
            b = a.transpose()
            diamonds = pd.concat([diamonds, b], ignore_index=True)

        self.df = diamonds
        return diamonds


class DriverScrapper(Scrapper):
    def __init__(self, url=None, driver_class='chrome'):
        super().__init__(url)
        self.url = url
        self.raw = None
        self.system = platform.system()

        self.driver_path = os.path.abspath("./{}driver_{}".format(driver_class, self.system))
        self.driver = webdriver.Chrome(self.driver_path)

    def get(self, num_pagedowns=50):
        self.driver.get(self.url)
        time.sleep(1)

        elem = self.driver.find_element_by_tag_name("body")

        while num_pagedowns:
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.2)
            num_pagedowns -= 1

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        self.raw = soup

        headerinctags = soup.find_all('div', class_='grid-header normal-header')
        header = headerinctags[0].get_text(';').split(";")[1:]
        diamondsmessy = soup.find_all('a', class_='grid-row row')

        # Start to construct DataFrame
        diamonds = pd.DataFrame(columns=header)

        for i in range(len(diamondsmessy)):
            a = diamondsmessy[i].get_text(";")
            b = a.split(";")
            del b[4]
            a = pd.DataFrame(b, index=header)
            b = a.transpose()
            diamonds = pd.concat([diamonds, b], ignore_index=True)

        self.driver.quit()

    def get_by_filter(self):
        pass