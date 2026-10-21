import pandas as pd
from mip.config import Config
import matplotlib.pyplot as plt
import seaborn as sns

config = Config()

# https://www.kaggle.com/datasets/karansarpal/fremtpl2-french-motor-tpl-insurance-claims
df_fre = pd.read_parquet(config.DATA_DIR / "frequency.parquet")

"""
- *IDpol*: **Versicherungsnummer** - The policy ID (used to link with the claims dataset).
- *ClaimNb*: **Anzahl der Schadensfälle** - Number of claims during the exposure period.
- *Exposure*: **Versicherungsdauer** - The period of exposure for a policy, in years.
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
- *ClaimAmount*: **Schadenshöhe** - Die Höhe des gemeldeten Schadens bzw. der vom Versicherer zu zahlende Betrag.
"""

def get_no_claims_percentage():
    """
    Calculate the percentage of policies with no claims.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the insurance data.
    
    Returns:
    float: The percentage of policies with no claims.
    """
    claim_dist = df_fre["ClaimNb"].value_counts(normalize=True).sort_index() * 100
    print(f'{claim_dist[0]:.2f}% of policies have no claims.')
    return claim_dist[0]

def see_exposure_distribution():
    """
    Visualize the distribution of the 'Exposure' variable.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the insurance data.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df_fre["Exposure"], bins=30, kde=True)
    plt.title("Distribution of Exposure")
    plt.xlabel("Exposure (years)")
    plt.ylabel("Frequency")
    plt.savefig(config.FIG_DIR / "exposure_distribution.png")


if __name__ == "__main__":
    # Calculate and print the percentage of policies with no claims
    no_claims_percentage = get_no_claims_percentage()
    
    # Visualize the distribution of the 'Exposure' variable
    see_exposure_distribution()

