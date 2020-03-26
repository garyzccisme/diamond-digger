import os
import platform
import time

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
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
        self.df_columns = None
        self.df = None

    def get(self):
        s = requests.session()
        BeautifulSoup(s.get(self.url, headers=self.headers).content, "html.parser")

        headerinctags = self.raw.find_all('div', class_='grid-header normal-header')
        header = headerinctags[0].get_text(';').split(";")[1:]
        diamondsmessy = self.raw.find_all('a', class_='grid-row row')

        # Start to construct the DataFrame
        diamonds = pd.DataFrame(columns=header)
        self.df_columns = header

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
        self.df = None
        self.df_columns = None
        self.system = platform.system()

        self.driver_path = os.path.abspath("./{}driver_{}".format(driver_class, self.system))
        if driver_class == 'chrome':
            self.driver = webdriver.Chrome(self.driver_path)

    def get(self, num_pages=50):
        self.driver.get(self.url)
        time.sleep(1)

        elem = self.driver.find_element_by_tag_name("body")

        while num_pages:
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.2)
            num_pages -= 1

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        headerinctags = soup.find_all('div', class_='grid-header normal-header')
        header = headerinctags[0].get_text(';').split(";")[1:]
        diamondsmessy = soup.find_all('a', class_='grid-row row')

        # Start to construct DataFrame
        diamonds = pd.DataFrame(columns=header)
        self.df_columns = header

        for i in range(len(diamondsmessy)):
            a = diamondsmessy[i].get_text(";")
            b = a.split(";")
            del b[4]
            a = pd.DataFrame(b, index=header)
            b = a.transpose()
            diamonds = pd.concat([diamonds, b], ignore_index=True)
        self.df = diamonds
        self.driver.quit()
        return diamonds

    def get_by_filter(self, carat_input=None, price_input=None,
                      pages_visited=25, scroll_number=30, scroll_pause_time=0.2):

        if carat_input is None:
            carat_input = [0.8, 1.0]
        if price_input is None:
            price_input = [1000, 5000]

        self.driver.get(self.url)

        # Set up carat filter
        # Use try to avoid Stale Element Reference Exception, should have better method
        try:
            self.set_filter_by_element_name(element_name='carat-min-input', value=carat_input[0])
        except:
            self.set_filter_by_element_name(element_name='carat-min-input', value=carat_input[0])
        try:
            self.set_filter_by_element_name(element_name='carat-max-input', value=carat_input[1])
        except:
            self.set_filter_by_element_name(element_name='carat-max-input', value=carat_input[1])

        # Cut up price range
        price_cuts = np.split(np.arange(*price_input), pages_visited)
        min_price_cuts = np.min(price_cuts, 1)
        max_price_cuts = np.max(price_cuts, 1)

        # Start to construct DataFrame
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        headerinctags = soup.find_all('div', class_='grid-header normal-header')
        header = headerinctags[0].get_text(';').split(";")[1:]

        diamonds = pd.DataFrame(columns=header)
        self.df_columns = header

        # Start scroll loop
        for p in range(pages_visited):
            self.driver.execute_script("window.scrollTo(0, 0)")

            # Update price filter
            try:
                self.set_filter_by_element_name(element_name='price-min-input', value=price_input[0])
            except:
                self.set_filter_by_element_name(element_name='price-min-input', value=price_input[0])
            try:
                self.set_filter_by_element_name(element_name='price-max-input', value=price_input[1])
            except:
                self.set_filter_by_element_name(element_name='price-max-input', value=price_input[1])

            # Scroll down
            elem = self.driver.find_element_by_tag_name("body")
            for scroll in range(scroll_number):
                elem.send_keys(Keys.PAGE_DOWN)
                time.sleep(scroll_pause_time)

            # Scrape data
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            diamondsmessy = soup.find_all('a', class_='grid-row row')

            for i in range(len(diamondsmessy)):
                a = diamondsmessy[i].get_text(";")
                b = a.split(";")
                del b[4]
                a = pd.DataFrame(b, index=header)
                b = a.transpose()
                diamonds = pd.concat([diamonds, b], ignore_index=True)

            self.df = diamonds
            return diamonds

    def set_filter_by_element_name(self, element_name=None, value=None):
        element = self.driver.find_element_by_name(element_name)
        element.click()
        element.send_keys(''.format(value))
        element.send_keys(Keys.ENTER)

