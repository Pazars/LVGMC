import os
import yaml
import pathlib
import pandas as pd
from pandas import DataFrame


def get_file_dir() -> pathlib.Path:
    """Get Path object to the directory this file is in.

    Returns:
        pathlib.Path: Path object
    """
    return pathlib.Path(__file__).resolve(strict=True).parent


def get_data_path(name) -> str:
    """Get the full CSV data file path.

    Assumes file is located in ./data/<name>

    Args:
        name (str): Data file name.

    Returns:
        str: Data file path
    """

    dir = os.path.dirname(__file__)
    return os.path.join(dir, 'data', name)


def get_config_dir() -> pathlib.Path:
    """Get the full CSV data file path.

    Assumes file is located in ./data/<name>

    Args:
        name (str): Data file name.

    Returns:
        pathlib.Path: Path object to configs directory
    """

    dir = get_file_dir()
    cfg_dir = dir / 'configs'
    return cfg_dir


def load_yaml_config_file(name: str) -> dict:
    """Load configuration yaml file.

    Args:
        name (str, optional): Configuration file name

    Returns:
        dict: YAML configuration file content in a dictionary.
    """

    dir = get_config_dir()
    path = dir / name
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_default_leap_year(default: int = 1900) -> int:
    """Find a dummy leap year to assign to the dataset.

    Default year used is 1900, but this is not a leap year.
    Parsing data will fail at February 29th.

    Args:
        default (int, optional): Default year used by Pandas. Defaults to 1900.

    Returns:
        int: Default leap year
    """

    year = default + 1
    ts = pd.Timestamp(year, 1, 1)
    while not ts.is_leap_year:
        year += 1
        ts = pd.Timestamp(year, 1, 1)
    return year


def load_year_data(cfg: dict) -> DataFrame:
    """Load data specified for a single year.

    Assumes cfg contains following keys:
        - data_fname: Dataset file name (only name, file in ./data)
        - datetime_column_name: Name of the datetime data column
        - datetime_format: Format of the datetime data
        - leap_year: Is the data for a leap year? Relevant only if
        year not given

    Args:
        cfg (dict): Configuration object

    Returns:
        DataFrame: Data loaded as a Pandas DataFrame
    """

    name = cfg['data_fname']
    dpath = get_data_path(name)
    df = pd.read_csv(dpath)

    dt_col_name = cfg['datetime_column_name']
    dt_format = cfg['datetime_format']

    if '%Y' or '%y' not in dt_format:
        # Exact year not specified in data

        if cfg['leap_year']:
            # Data is of a leap year
            def_year = get_default_leap_year()

            dt_format = f'%Y.{dt_format}'

            df[dt_col_name] = pd.to_datetime(
                f'{def_year}.' + df[dt_col_name],
                format=dt_format,
            )

    else:
        df[dt_col_name] = pd.to_datetime(
            df[dt_col_name],
            format=dt_format,
        )

    return df


def binary_search(arr, low, high, x):

    # print(f"low index: {low}, high index: {high}")
    # print(f"low value: {arr[low]}, high value: {arr[high]}")

    if abs(high - low) == 1:
        return (min(low, high), max(low, high))

    else:

        # Check base case
        if high >= low:

            mid = (high + low) // 2

            # If element is present at the middle itself
            if arr[mid] == x:
                return mid

            # If element is smaller than mid, then it can only
            # be present in left subarray
            elif arr[mid] > x:
                return binary_search(arr, low, mid - 1, x)

            # Else the element can only be present in right subarray
            else:
                return binary_search(arr, mid + 1, high, x)

        else:
            # Element is not present in the array
            return -1
