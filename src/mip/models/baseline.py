# Baseline Model Evaluation
# Ignore Area, VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Density, Region
from sklearn.metrics import mean_poisson_deviance

def baseline_deviance(train, test):
    """
    Calculate the baseline frequency and Poisson deviance for the test set.

    Parameters:
    train (pd.DataFrame): The training DataFrame.
    test (pd.DataFrame): The testing DataFrame.

    Returns:
    None
    """
    baseline_frequency = (
        train["ClaimNb"].sum() / train["Exposure"].sum()
    )

    y_true = test["ClaimNb"]
    y_pred = baseline_frequency * test["Exposure"]

    deviance = mean_poisson_deviance(y_true, y_pred)

    return deviance, baseline_frequency
