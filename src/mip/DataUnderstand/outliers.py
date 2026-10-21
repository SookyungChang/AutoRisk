import pandas as pd
from mip.config import Config
config = Config()

df = pd.read_parquet(config.DATA_DIR / "frequency.parquet")


def calculate_whiskers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    lower_whisker = df.loc[df[column] >= lower_bound, column].min()
    upper_whisker = df.loc[df[column] <= upper_bound, column].max()

    print("Lower whisker:", lower_whisker)
    print("Upper whisker:", upper_whisker)
    print("Lower outlier boundary:", lower_bound)
    print("Upper outlier boundary:", upper_bound)

    return lower_whisker, upper_whisker, lower_bound, upper_bound

