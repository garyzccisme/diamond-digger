import os
import platform
import time
from typing import Dict, List, Union

import bs4
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
from selenium import webdriver
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.keys import Keys


class BlueNileScrapper:
    """
    Superclass of Scrapper(currently only available for https://www.bluenile.com/diamond-search).
    Will consider for other retailer's scrapper in the future.
    """

    def __init__(self, url: str):
        self.url = url
        self.soup = None
        self.df = None

    def get(self):
        raise NotImplementedError

    def get_column_name(self, soup: BeautifulSoup = None, class_name: str = 'grid-header normal-header') -> List:
        """
        Load column name from website, should be originally 17 columns. Usually not got called separately.
        Notes: 'Stock No.' is primary key to identify diamonds. Others are well explained in the Internet and also
        covered in the README file.

        Args:
            soup: Raw material scrapped by BeautifulSoup from website. Default is None but assigned by self.soup.
            class_name: The HTML class name containing column names, always use the default one.

        Returns: column name list
        ['Shape', 'Price', 'Discount Price', 'Carat', 'Cut', 'Color', 'Clarity', 'Polish', 'Symmetry',
         'Fluorescence', 'Depth', 'Table', 'L/W', 'Price/Ct', 'Culet', 'Stock No.', 'Delivery Date']

        """
        # For stability, we use hard coded column name here
        return ['Shape', 'Price', 'Discount Price', 'Carat', 'Cut', 'Color', 'Clarity', 'Polish', 'Symmetry',
                'Fluorescence', 'Depth', 'Table', 'L/W', 'Price/Ct', 'Culet', 'Stock No.', 'Delivery Date']

        # if soup is None:
        #     soup = self.soup
        #
        # column_name = soup.find_all('div', class_=class_name)[0].get_text(';').split(";")
        #
        # # Need Manually Check! Headers will change!
        # print(column_name)
        # # column_name[0] is 'Wish List'
        # del column_name[0]
        # # Insert new column 'Discount Price'
        # column_name.insert(2, 'Discount Price')
        # return column_name

    def get_record(self, soup: BeautifulSoup = None, class_name: str = 'grid-row row TL511DiaStrikePrice') -> List:
        """
        Load data from website. Usually not got called separately.

        Args:
            soup: Raw material scrapped by BeautifulSoup from website. Default is None but assigned by self.soup.
            class_name: The HTML class name containing column names, always use the default one.

        Returns: list of data records

        """
        if soup is None:
            soup = self.soup

        diamond_mine = soup.find_all('a', class_=class_name)
        diamond_list = []
        for raw_diamond in diamond_mine:
            finished_diamond = self.detect_discount(raw_diamond)
            diamond_list.append(finished_diamond)
        return diamond_list

    def detect_discount(self, element: bs4.element.Tag) -> List:
        """
        There're records having discount while others not. Thus need to take additional detection to make data
        consistency.

        Returns: single record

        """
        record = element.get_text(';').split(';')
        # Check if the diamond has discount price, if not will copy the origin price as discount price
        if record[1] == 'Was: ':
            del record[1]
            del record[2]
        else:
            record.insert(2, record[1])
        del record[4]
        return record


class RequestsBlueNileScrapper(BlueNileScrapper):
    """
    Child class of BlueNileScrapper engined by requests. Since BlueNile website is infinite loading, requests can
    only scrape the current window, the scrape result is limited. Can use this to get a dataset sample.

    headers = {
        "Referer": "https://www.bluenile.com/diamonds/round-cut",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.87 Safari/537.36",
        "Host": "www.bluenile.com",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
    }
    """

    def __init__(self, url: str = None, headers: Dict = None):
        super().__init__(url)
        self.headers = headers

    def get(self) -> pd.DataFrame:
        """
        Load DataFrame from website.

        Returns: pd.DataFrame

        """
        s = requests.session()
        self.soup = BeautifulSoup(s.get(self.url, headers=self.headers).content, "html.parser")
        column_name = self.get_column_name()
        diamond_list = self.get_record()
        self.df = pd.DataFrame(diamond_list, columns=column_name)
        return self.df


class DriverBlueNileScrapper(BlueNileScrapper):
    """
    Child class of BlueNileScrapper engined by selenium.webdriver, can load completed data by controlling web driver.
    """

    def __init__(self, url=None, driver_class='chrome'):
        """
        Args:
            url: Web url, should manually input 'https://www.bluenile.com/diamond-search'.
            driver_class: Use chrome driver to scrap, different system has it's own driver.
        """
        super().__init__(url)
        self.driver = None
        self.driver_class = driver_class
        # Find correct driver absolute path
        self.driver_path = os.path.abspath("./{}driver_{}".format(driver_class, platform.system()))
        self.soup_list = []

    def _launch_driver(self):
        if self.driver is None:
            if self.driver_class == 'chrome':
                self.driver = webdriver.Chrome(self.driver_path)
            self.driver.get(self.url)
            time.sleep(1)

    def _quit_driver(self):
        self.driver.quit()
        self.driver = None

    def _scroll(self, scroll_number: int = None, scroll_pause_time: int = None):
        """
        Scroll down command for driver.
        If scroll_number is given, then do given times of scrolling.
        If not, then do scrolling until nothing more to load.

        Args:
            scroll_number: The number of scrolling times.
            scroll_pause_time: The pause time (second) for each scrolling.
        """
        if scroll_number:
            if scroll_pause_time is None:
                scroll_pause_time = 0.5
            window = self.driver.find_element_by_tag_name("body")
            for scroll in range(scroll_number):
                window.send_keys(Keys.PAGE_DOWN)
                time.sleep(scroll_pause_time)
        else:
            if scroll_pause_time is None:
                scroll_pause_time = 10
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            while True:
                # Scroll down to bottom
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause_time)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                # Check if there's anything to load
                if new_height == last_height:
                    break
                last_height = new_height

    def get(self, carat_input: List = None, price_input: List = None,
            scroll_number: int = None, scroll_pause_time: int = None,
            is_quit: bool = True, return_df: bool = True) -> Union[pd.DataFrame, List]:
        """
        Load DataFrame by setting fixed filters for carat and price.

        Args:
            carat_input: [Min Carat, Max Carat]. Shouldn't exceed [0.23, 20.98].
            price_input: [Min Price, Max Price]. Shouldn't exceed [261, 1860430].
            scroll_number: The number of scrolling times.
            scroll_pause_time: The pause time (second) for each scrolling.
            is_quit: If True then quit browser after finish, else keep it.
            return_df: If True then return final DataFrame, else return records(List).

        Returns: pd.DataFrame

        """
        if carat_input is None:
            carat_input = [0.23, 20.98]
        if price_input is None:
            price_input = [261, 1860430]

        # Launch web driver
        self._launch_driver()

        # Set filter
        self._set_filter_by_element_name(element_name='carat-max-input', value=carat_input[1])
        self._set_filter_by_element_name(element_name='carat-min-input', value=carat_input[0])

        self._set_filter_by_element_name(element_name='price-max-input', value=price_input[1])
        self._set_filter_by_element_name(element_name='price-min-input', value=price_input[0])

        # Scroll down
        self._scroll(scroll_number=scroll_number, scroll_pause_time=scroll_pause_time)

        # Scrape
        self.soup = BeautifulSoup(self.driver.page_source, "html.parser")
        column_name = self.get_column_name()
        diamond_list = self.get_record()
        self.df = pd.DataFrame(diamond_list, columns=column_name)

        if is_quit:
            self._quit_driver()
        else:
            self.driver.execute_script("window.scrollTo(0, 0)")

        if return_df:
            return self.df
        else:
            return diamond_list

    def get_dynamic(self, carat_set: List = None, price_set: List = None,
                    scroll_number: int = None, scroll_pause_time: int = None,
                    keep_soup_list: bool = False) -> pd.DataFrame:
        """
        Load DataFrame by setting dynamic filters for carat and price. Used for production.

        Args:
            carat_set: [[carat_min_1, carat_max_1], [carat_min_2, carat_max_2], ...], should be ascending.
            price_set: [[price_min_1, price_max_1], [price_min_2, price_max_2], ...], should be ascending.
            scroll_number: The number of scrolling times.
            scroll_pause_time: The pause time (second) for each scrolling.
            keep_soup_list: If True then store all pages' soup into self.soup_list, else not.

        Returns: DataFrame

        """
        if carat_set is None:
            carat_set = [[0.23, 20.98]]
        if price_set is None:
            price_set = [[261, 1860430]]

        if len(carat_set) != len(price_set):
            raise ValueError("carat_set and price_set should have same length!")

        # Launch web driver
        self._launch_driver()

        # Scrap column name
        self.soup = BeautifulSoup(self.driver.page_source, "html.parser")
        column_name = self.get_column_name()

        # Overwrite self.soup, each page's soup will be stored in list
        diamond_list = []

        pages = len(carat_set)
        for page in range(pages):
            carat_filter = carat_set[page]
            price_filter = price_set[page]

            # Input each set of filter into self.get()
            diamond_list += self.get(carat_input=carat_filter, price_input=price_filter,
                                     scroll_number=scroll_number, scroll_pause_time=scroll_pause_time,
                                     is_quit=False, return_df=False)
            if keep_soup_list:
                self.soup_list.append(self.soup)

        self.df = pd.DataFrame(diamond_list, columns=column_name)

        self._quit_driver()
        return self.df

    def diamond_distribution_research(self, carat_set: List = None, price_set: List = None) -> pd.DataFrame:
        """
        This method is used to count total diamonds number given by specific filter set.
        Is usually used for diamond distribution research.
        Args:
            carat_set: [[carat_min_1, carat_max_1], [carat_min_2, carat_max_2], ...], should be ascending.
            price_set: [[price_min_1, price_max_1], [price_min_2, price_max_2], ...], should be ascending.

        Returns: DataFrame | carat_filter | price_filter | count |

        """
        if carat_set is None:
            carat_set = [[0.23, 20.98]]
        if price_set is None:
            price_set = [[261, 1860430]]

        if len(carat_set) != len(price_set):
            raise ValueError("carat_set and price_set should have same length!")

        # Launch web driver
        self._launch_driver()
        all_diamonds_list = []

        pages = len(carat_set)
        for page in range(pages):
            carat_filter = carat_set[page]
            price_filter = price_set[page]

            # Set filter
            self._set_filter_by_element_name(element_name='carat-max-input', value=carat_filter[1])
            self._set_filter_by_element_name(element_name='carat-min-input', value=carat_filter[0])

            self._set_filter_by_element_name(element_name='price-max-input', value=price_filter[1])
            self._set_filter_by_element_name(element_name='price-min-input', value=price_filter[0])

            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            all_diamonds = soup.find_all(
                'div', class_='navigation-tabs sticky filter-tooltip-cta')[0].get_text(';').split(';')[2]
            all_diamonds_list.append(int(all_diamonds.replace(',', '')))

        self._quit_driver()
        distribution_df = pd.DataFrame(np.array([carat_set, price_set, all_diamonds_list]).T,
                                       columns=['carat', 'price', 'count'])

        return distribution_df

    def _set_filter_by_element_name(self, element_name: str = None, value: float = None, click_pause_time: int = 1):
        """
        Helper function to find and set specific filter.

        Args:
            element_name: HTML element name of filter.
            value: The input value of filter.
            click_pause_time: The pause time (second) for each click

        """
        # Use try & except to avoid StaleElementReferenceException, should have better method
        # https://stackoverflow.com/questions/27003423/staleelementreferenceexception-on-python-selenium
        try:
            element = self.driver.find_element_by_name(element_name)
            time.sleep(click_pause_time)
            element.click()
            time.sleep(click_pause_time)
            element.send_keys('{}'.format(value))

        except StaleElementReferenceException:
            element = self.driver.find_element_by_name(element_name)
            time.sleep(click_pause_time)
            element.click()
            time.sleep(click_pause_time)
            element.send_keys('{}'.format(value))

        element.send_keys(Keys.ENTER)
        time.sleep(click_pause_time)
