import logging
import os
from datetime import date
from datetime import datetime
from typing import List

import pandas as pd

from DiamondScrapper.Scrapper import DriverBlueNileScrapper

logging.basicConfig(filename='./Data/log.txt', filemode='a', format='%(asctime)s %(message)s', level=logging.INFO)


def auto_scrape_pipline(driver_class='chrome', url='https://www.bluenile.com/diamond-search',
                        carat_set: List = None, price_set: List = None,
                        save_single_pkl: bool = True, set_name: str = None):
    logging.info('\n')
    logging.info('================ Start Scrapping ===============')

    today = date.today()
    scrapper = DriverBlueNileScrapper(url=url, driver_class=driver_class)
    df = scrapper.get_dynamic(carat_set=carat_set, price_set=price_set)

    logging.info('===== Finish Scrapping =====')

    # Drop Duplicates
    df.drop_duplicates(inplace=True)

    # Transform DataFrame
    df = transformation(df)

    logging.info('===== Finish Transformation =====')

    # Save today's single df
    if save_single_pkl:
        save_pkl(df, './Data/{}/blue_niles_df_{}.pkl'.format(today.strftime('%Y_%m_%d'), set_name))

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
    logging.info('==================== Finish ====================')


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
    update_column = ['Price', 'Discount Price', 'Price/Ct', 'Delivery Date', 'Last Available Date']

    existing_index = set(df.index) & set(main_df.index)
    main_df.loc[existing_index, update_column] = df.loc[existing_index, update_column]
    logging.info('===== {} records updated ====='.format(len(existing_index)))

    # Add new records
    new_records = df[~df.index.isin(existing_index)]
    main_df = pd.concat([main_df, new_records])
    logging.info('===== {} new records ====='.format(new_records.shape[0]))

    if is_save:
        save_pkl(main_df, main_df_path)
    else:
        return main_df


###### Hard Code Filters #####

Carat = [0.23, 20.98]
Price = [261, 1860430]

carat_range = {
    'carat_range_05_052': [[0.5, 0.52]] * 19,
    'carat_range_053_059': [*[[0.53, 0.54]] * 3, [0.55, 0.55], *[[0.56, 0.59]] * 2],
    'carat_range_06_099': [*[[0.6, 0.6]] * 4,
                           [0.61, 0.61], [0.62, 0.69], *[[0.7, 0.7]] * 6,
                           [0.71, 0.71], [0.72, 0.74], [0.75, 0.79], *[[0.8, 0.8]] * 3,
                           [0.81, 0.89], *[[0.9, 0.9]] * 3, [0.91, 0.99]],
    'carat_range_1_101': [[1, 1.01]] * 11,
    'carat_range_102_122': [[1.02, 1.02], *[[1.03, 1.22]] * 7],
    'carat_range_123': [[1.23, 1.29], [1.30, 1.32], [1.33, 1.49], [1.5, 1.5], [1.51, 1.59], [1.6, 2.0],
                        [2.01, 2.01], [2.02, 2.23], [2.24, 3], [3, 3.99], [4, 20.98]],
}

price_range = {
    'price_range_05_052': [[261, 780], [781, 850], [851, 900], [901, 950], [951, 990], [991, 1030], [1031, 1065],
                           [1066, 1100], [1101, 1130], [1131, 1165], [1166, 1200], [1201, 1245], [1246, 1275],
                           [1276, 1305], [1306, 1365], [1366, 1455], [1456, 1580], [1581, 1800], [1801, 1860430]],
    'price_range_053_059': [[261, 1150], [1151, 1400], [1401, 1860430], Price, [261, 1350], [1351, 1860430]],
    'price_range_06_099': [[261, 1350], [1351, 1650], [1651, 2000], [2001, 1860430],
                           *[Price] * 2, [261, 1600], [1601, 1850], [1851, 2100], [2101, 2350], [2351, 2900],
                           [2901, 1860430], *[Price] * 3, [261, 2600], [2601, 3300], [3301, 1860430],
                           Price, [261, 3400], [3401, 4400], [4401, 1860430], Price],
    'price_range_1_101': [[261, 3350], [3351, 3750], [3751, 4050], [4051, 4400], [4401, 4750], [4751, 5100],
                          [5101, 5450], [5451, 5800], [5801, 6300], [6301, 7000], [7001, 1860430]],
    'price_range_102_122': [Price, [261, 4800], [4801, 5550], [5551, 6150], [6151, 6750], [6751, 7500], [7501, 8700],
                            [8701, 1860430]],
    'price_range_123': [Price] * 11,
}

if __name__ == "__main__":
    # Total hard coded filter sets are 6
    logging.info("\n\n\nToday is {} \n".format(str(date.today())))
    for i in range(6):
        carat_set_name = list(carat_range.keys())[i]
        price_set_name = list(price_range.keys())[i]

        try:
            auto_scrape_pipline(carat_set=carat_range[carat_set_name], price_set=price_range[price_set_name],
                                set_name=carat_set_name)
        except:
            logging.info('filter set {}, {} BREAK!!!'.format(carat_set_name, price_set_name))
            logging.info('TRY AGAIN')
            try:
                auto_scrape_pipline(carat_set=carat_range[carat_set_name], price_set=price_range[price_set_name],
                                    set_name=carat_set_name)
            except:
                logging.info(
                    'filter set {}, {} BREAK AGAIN!!! REQUIRE MANUAL CHECK!!!'.format(carat_set_name, price_set_name))
            continue

        logging.info('=====Finish filter set {}, {}====='.format(carat_set_name, price_set_name))



