import pandas as pd
from IPython.display import display


def show(df: pd.DataFrame, limit: int = 5) -> None:
    print(df.shape)
    display(df.head(limit))


def show_null_percentages(df: pd.DataFrame) -> None:
    nulls = df.isnull().sum() / df.shape[0]
    nulls = nulls.sort_values(ascending=False)
    display(nulls)


def count_frequent_values(series: pd.Series, min_count: int = 10) -> int:
    """
    Count the number of unique values in a series that appear more than a
    specified number of times.

    Parameters:
    series (pd.Series): The pandas Series to count values from.
    min_count (int): The minimum number of occurrences for a value to be
    counted. Default is 10.

    Returns:
    int: The number of unique values that appear more than min_count times.
    """
    vc = series.value_counts()
    vc = vc[vc > min_count]
    return len(vc)
