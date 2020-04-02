from datetime import date
from datetime import datetime
import logging
import os
from typing import List

import pandas as pd

from DiamondScrapper.Scrapper import DriverBlueNileScrapper


logging.basicConfig(filename='./Data/log.txt', filemode='a', format='%(asctime)s %(message)s', level=logging.INFO)


def auto_scrape_pipline(driver_class='chrome', url='https://www.bluenile.com/diamond-search',
                        carat_input: List = None, price_input: List = None):
    logging.info('\n')
    logging.info('========== Start Scrapping ==========')
    logging.info('Filter Condition: carat_input={}, price_input={}'.format(carat_input, price_input))

    today = date.today()
    scrapper = DriverBlueNileScrapper(url=url, driver_class=driver_class)
    df = scrapper.get_dynamic(carat_input=carat_input, price_input=price_input)

    logging.info('===== Finish Scrapping =====')

    # Transform DataFrame
    df = transformation(df)

    logging.info('===== Finish Transformation =====')

    # Save today's df
    save_pkl(df, './Data/blue_niles_df_{}.pkl'.format(str(today)))

    logging.info('===== Finish save =====')

    # Add new columns for update
    length = df.shape[0]
    df['Last Available Date'] = [today] * length
    df['First Available Date'] = [today] * length

    # Update DataFrame to main DataFrame
    logging.info('===== Start update =====')

    if os.path.isfile('./Data/blue_niles_df.pkl'):
        update(df)
    else:
        save_pkl(df, './Data/blue_niles_df.pkl')

    logging.info('===== Finish update and save =====')
    logging.info('========== Finish ==========')


def save_pkl(df, path=None):
    df.to_pickle(path)


def save_csv(df, path=None):
    df.to_csv(path)


def pkl2csv(pkl_path, csv_path):
    df = pd.read_pickle(pkl_path)
    save_csv(df, csv_path)


def transformation(df):

    df.set_index('Stock No.', inplace=True)

    df['Price'] = df['Price'].apply(lambda x: int(x[1:].replace(',', '')))
    df['Discount Price'] = df['Discount Price'].apply(lambda x: int(x[1:].replace(',', '')))
    df['Price/Ct'] = df['Price/Ct'].apply(lambda x: int(x[1:].replace(',', '')))

    df = df.astype({'Carat': 'float', 'Depth': 'float', 'Table': 'float', 'L/W': 'float'})

    df['Delivery Date'] = df['Delivery Date'].apply(find_year)

    return df


def find_year(day: str):
    today = date.today()
    this_year = datetime.strptime(day + ' {}'.format(today.year), '%b %d %Y').date()
    if today <= this_year:
        return this_year
    else:
        return datetime.strptime(day + ' {}'.format(today.year + 1), '%b %d %Y').date()


def update(df, main_df_path='./Data/blue_niles_df.pkl', is_save=True):

    main_df = pd.read_pickle(main_df_path)

    # Update values for existing records
    main_df.update(df, join='left')
    logging.info('===== {} records updated ====='.format(len(set(df.index) & set(main_df))))

    # Add new records
    new_records = df[df.index.isin(set(df.index) - set(main_df))]
    main_df = pd.concat([main_df, new_records])
    logging.info('===== {} new records ====='.format(new_records.shape[0]))

    if is_save:
        save_pkl(main_df, main_df_path)
    else:
        return main_df


if __name__ == "__main__":
    auto_scrape_pipline(carat_input=[0.8, 1.2], price_input=[1300, 18000])

