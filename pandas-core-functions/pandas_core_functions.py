import pandas as pd
import seaborn as sns
import logging as logger

from matplotlib import pyplot as plt
from typing import Tuple

logger.basicConfig(level=logger.INFO)


def graph_regression_plot(df_merge: pd.DataFrame) -> None:
    sns.set(rc={'figure.figsize': (12, 6)})
    sns.jointplot(x='Rainfall', y='Temperature', data=df_merge, kind='reg')
    plt.show()


def sort_data(df_merge: pd.DataFrame) -> None:
    df_sorted_by_temp = df_merge.sort_values(by='Temperature', ascending=False)
    logger.info(f'Sorted by temperature in descending order: {df_sorted_by_temp}')
    df_sorted_by_rainfall = df_merge.sort_values(by='Rainfall')
    logger.info(f'Sorted by rainfall in ascending order: {df_sorted_by_rainfall}')


def merge_data(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    df_outer: pd.DataFrame = pd.merge(df_1, df_2, on='Year', how='outer')
    logger.info(f'Outer join: {df_outer}')
    df_inner: pd.DataFrame = pd.merge(df_1, df_2, on='Year', how='inner')
    logger.info(f'Inner join: {df_inner}')
    return df_inner


def filter_data(df_temp: pd.DataFrame, df_rain: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_temp_f: pd.DataFrame = df_temp.query('Temperature > 0 & Temperature < 40')
    logger.info(f'Filtered by temperature: {df_temp_f}')
    df_temp_f.plot.scatter(x='Year', y='Temperature', label='Temperature and Year')
    plt.show()

    df_rain_f: pd.DataFrame = df_rain.query('Rainfall > 0 & Rainfall < 6')
    logger.info(f'Filtered by rainfall: {df_rain_f}')
    df_rain_f.plot.scatter(x='Year', y='Rainfall', label='Rainfall and Year')
    plt.show()
    return df_temp_f, df_rain_f


def read_data(dataset_1: str, dataset_2: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_temp: pd.DataFrame = pd.read_csv(dataset_1)
    logger.info(f'Read from temperature dataset: {df_temp}')
    df_temp.plot.scatter(x='Year', y='Temperature', label='Temperature and Year')
    plt.show()

    df_rain: pd.DataFrame = pd.read_csv(dataset_2)
    logger.info(f'Read from rainfall dataset: {df_rain}')
    df_rain.plot.scatter(x='Year', y='Rainfall', label='Rainfall and Year')
    plt.show()
    return df_temp, df_rain


def main():
    # Read data and plot
    df_temp, df_rain = read_data('temperature_yearly.csv', 'rain_yearly.csv')

    # Filter data and plot
    df_temp_f, df_rain_f = filter_data(df_temp, df_rain)

    # Merge data
    df_merge: pd.DataFrame = merge_data(df_temp_f, df_rain_f)

    # Sort data
    sort_data(df_merge)

    # Graph linear regression
    graph_regression_plot(df_merge)


if __name__ == "__main__":
    main()
