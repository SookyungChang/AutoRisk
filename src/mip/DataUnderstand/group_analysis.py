import pandas as pd
import numpy as np
from mip.config import Config
import matplotlib.pyplot as plt
import seaborn as sns

config = Config()

df = pd.read_parquet(config.DATA_DIR / "frequency.parquet")

def grouping(df: pd.DataFrame, group_column: str,  bins: list = None, labels: list = None, q: int = 5) -> pd.DataFrame:
    if bins is None:
        return pd.qcut(df[group_column], q=q)
    else:
        return pd.cut(df[group_column], bins=bins, labels=labels)
        

def take_logrithm(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    return np.log1p(df[column_name])

def group_analysis(df: pd.DataFrame, new_column_name: str) -> pd.DataFrame:
    grouped = df.groupby(new_column_name).agg(
            total_policies=('ClaimNb', 'sum'),
            total_exposure=('Exposure', 'sum'),
        ).reset_index()
    grouped['claim_frequency'] = grouped['total_policies'] / grouped['total_exposure']

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        data=grouped,
        x=new_column_name,
        y="claim_frequency",
        color="steelblue",
        ax=ax
    )

    ax.set(
        title="Claim Frequency by " + new_column_name,
        xlabel=new_column_name,
        ylabel="Claims per Policy-Year"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(config.FIG_DIR / "group_analysis" / f"{new_column_name}_analysis.png")
    print(f"Saved figure: {config.FIG_DIR / 'group_analysis' / f'{new_column_name}_analysis.png'}")

if __name__ == "__main__":
    # Example usage of the bins_analysis function
    """
    - *Area*: **Geografisches Gebiet** - The density value of the city community where the car driver lives in: from "A" for rural area
    to "F" for urban centre.
    - *VehPower*: **Motorleistung** - The power of the car (ordered values).
    - *VehAge*: **Fahrzeugalter** - The vehicle age, in years.
    - *DrivAge*: **Alter des Fahrers** - The driver age, in years (in France, people can drive a car at 18).
    - *BonusMalus*: **Bonus-Malus-Klasse** - Bonus/malus, between 50 and 350: <100 means bonus, >100 means malus in France.
    - *VehBrand*: **Fahrzeugmarke** - The car brand (unknown categories)
    - *VehGas*: **Kraftstoffart** - The car gas, Diesel or regular.
    - *Density*: **Bevölkerungsdichte** - The density of inhabitants (number of inhabitants per square-kilometer) of the city where
    the car driver lives in.
    - *Region*: **Region** - The policy region in France (based on the 1970-2015 classification).
    """
    # df['logDensity'] = take_logrithm(df, column_name="Density")
    # print(df["BonusMalus"].dtype)
    # print(sorted(df["BonusMalus"].unique())[:20])

    # bins   = [49, 50, 90, 99, 100, 230]
    # labels = ["50", "51-90", "91-99", "100", "101+"]

    group_analysis(df, new_column_name="Region")





