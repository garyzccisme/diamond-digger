import os
import platform
import time

import bs4
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


class BlueNileScrapper:
    def __init__(self, url):
        self.url = url
        self.soup = None
        self.df = None

    def get(self):
        return

    def get_column_name(self, soup=None, class_name='grid-header normal-header'):

        if soup is None:
            soup = self.soup

        column_name = soup.find_all('div', class_=class_name)[0].get_text(';').split(";")
        del column_name[0]
        column_name.insert(2, 'Discount_Price')
        return column_name

    def get_record(self, soup=None, class_name='grid-row row TL511DiaStrikePrice'):

        if soup is None:
            soup = self.soup

        diamond_mine = soup.find_all('a', class_=class_name)
        diamond_list = []

        for raw_diamond in diamond_mine:
            finished_diamond = self.detect_discount(raw_diamond)
            diamond_list.append(finished_diamond)

        return diamond_list

    def detect_discount(self, element: bs4.element.Tag):
        record = element.get_text(';').split(';')
        if record[1] == 'Was: ':
            del record[1]
            del record[2]
        else:
            record.insert(2, record[1])
        del record[4]
        return record


class RequestsBlueNileScrapper(BlueNileScrapper):
    """
    headers = {"Referer": "https://www.bluenile.com/diamonds/round-cut",
           "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36",
           "Host": "www.bluenile.com",
           "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",}

    """
    def __init__(self, url=None, headers=None):
        super().__init__(url)
        self.headers = headers

    def get(self):
        s = requests.session()
        self.soup = BeautifulSoup(s.get(self.url, headers=self.headers).content, "html.parser")

        column_name = self.get_column_name()
        diamond_list = self.get_record()
        self.df = pd.DataFrame(diamond_list, columns=column_name)

        return self.df


class DriverBlueNileScrapper(BlueNileScrapper):
    def __init__(self, url=None, driver_class='chrome'):
        super().__init__(url)
        self.driver = None
        self.system = platform.system()
        self.driver_class = driver_class
        self.driver_path = os.path.abspath("./{}driver_{}".format(driver_class, self.system))

    def launch_driver(self):
        if self.driver_class == 'chrome':
            self.driver = webdriver.Chrome(self.driver_path)

    def quit_driver(self):
        self.driver.quit()

    def get(self, scroll_number=50, scroll_pause_time=0.2):
        self.launch_driver()
        self.driver.get(self.url)
        time.sleep(1)

        # Scroll down
        window = self.driver.find_element_by_tag_name("body")
        for scroll in range(scroll_number):
            window.send_keys(Keys.PAGE_DOWN)
            time.sleep(scroll_pause_time)

        self.soup = BeautifulSoup(self.driver.page_source, "html.parser")
        column_name = self.get_column_name()
        diamond_list = self.get_record()
        self.df = pd.DataFrame(diamond_list, columns=column_name)

        self.quit_driver()
        return self.df

    def get_by_dynamic_filter(self, carat_input=None, price_input=None,
                              pages_visited=5, scroll_number=30, scroll_pause_time=0.2):

        if carat_input is None:
            carat_input = [0.5, 0.8]
        if price_input is None:
            price_input = [1000, 5000]

        self.launch_driver()
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
        self.soup = BeautifulSoup(self.driver.page_source, "html.parser")
        column_name = self.get_column_name()

        # Since multiple pages will be visited, store all soups into list
        self.soup = []
        total_record = []

        # Start scroll loop
        for p in range(pages_visited):
            self.driver.execute_script("window.scrollTo(0, 0)")

            # Update price filter
            try:
                self.set_filter_by_element_name(element_name='price-min-input', value=min_price_cuts[p])
            except:
                self.set_filter_by_element_name(element_name='price-min-input', value=min_price_cuts[p])
            try:
                self.set_filter_by_element_name(element_name='price-max-input', value=max_price_cuts[p])
            except:
                self.set_filter_by_element_name(element_name='price-max-input', value=max_price_cuts[p])

            # Scroll down
            window = self.driver.find_element_by_tag_name("body")
            for scroll in range(scroll_number):
                window.send_keys(Keys.PAGE_DOWN)
                time.sleep(scroll_pause_time)

            # Scrape data
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            self.soup.append(soup)
            total_record += self.get_record(soup=soup)

        self.df = pd.DataFrame(total_record, columns=column_name)
        self.quit_driver()
        return self.df

    def set_filter_by_element_name(self, element_name=None, value=None):
        element = self.driver.find_element_by_name(element_name)
        element.click()
        time.sleep(3)
        element.send_keys('{}'.format(value))
        element.send_keys(Keys.ENTER)

