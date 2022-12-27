import helpers
from pandas import DataFrame


DEBUG = False


def main():

    cfg = helpers.load_yaml_config_file('task1_config.yaml')
    df = helpers.load_year_data(cfg)

    check_data(df)

    if DEBUG:
        print(df.head())
        print(df.dtypes)


def check_data(df: DataFrame):

    stations = ['x1', 'x2', 'x3']

    nrows = len(df.index)
    nans = df.isna()
    nans_sum = df.isna().sum()

    station_nans = nans[stations]
    print(station_nans[station_nans.all(axis=1)].sum())

    # print(nans.head())

    # nans[nans.all() == True]
    # print(nans[nans.all() == True].head())

    print(f"Number of data rows: {nrows}")

    for station in stations:
        nan = nans_sum[station]
        nanp = round(nan / nrows * 100, 2)
        print(f"Missing data rows for station '{station}': {nan} ({nanp}%)")


if __name__ == '__main__':
    main()
