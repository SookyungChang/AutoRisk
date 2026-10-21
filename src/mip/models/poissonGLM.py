import statsmodels.api as sm
from pathlib import Path
import pandas as pd
import numpy as np
from mip.DataUnderstand.group_analysis import grouping, take_logrithm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_poisson_deviance
from mip.models.baseline import baseline_deviance
from mip.config import Config
config = Config()

df = pd.read_parquet(config.DATA_DIR / "frequency.parquet")

def one_hot_encode(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    One-hot encode the specified categorical columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to encode.
    categorical_cols (list): List of categorical column names to encode.

    Returns:
    pd.DataFrame: The DataFrame with one-hot encoded columns.
    """
    df_encoded = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    train, test = train_test_split(
            df_encoded,
            test_size=config.TEST_SIZE,
            random_state=config.SEED
        )
    
    return train, test

def model_path(model_name: str) -> Path:
    return config.SAVED_MODELS_DIR / 'poissonGLM' / f"{model_name}.pkl"

def poisson_glm(train, categorical_cols: list, model_name: str):

    cols = [
    col for col in train.columns
    if col.startswith((tuple(name + "_" for name in categorical_cols)))
    ]

    model = sm.GLM(
        train["ClaimNb"],
        sm.add_constant(train[cols]),
        family=sm.families.Poisson(),
        offset=np.log(train["Exposure"])
    )

    results = model.fit()

    path = model_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.save(path)
    print(f"Saved model: {path}")

    return results

def offset_test(results, train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    mu = results.fittedvalues
    print("offset test for train set: ", mu.sum(), train["ClaimNb"].sum())

    exog_names = results.model.exog_names         
    X_test = sm.add_constant(test, has_constant="add")
    X_test = X_test.reindex(columns=exog_names, fill_value=0)

    mu_test = results.predict(X_test, offset=np.log(test["Exposure"]))
    mu_test = mu_test.rename("mu")
    print("offset test for test set: ", mu_test.sum(), test["ClaimNb"].sum())
    return mu_test


def measure_noise(mu, n_sim=100, seed=0):
    rng = np.random.default_rng(seed)    
    poisson = sm.families.Poisson()
    list_sim = []
    for _ in range(n_sim):
        y_sim = rng.poisson(mu)
        list_sim.append(mean_poisson_deviance(y_sim, mu))
    return np.array(list_sim)
    

if __name__ == "__main__":
    df['logDensity'] = take_logrithm(df, column_name="Density")
    for col in ["VehPower", "VehAge", "logDensity"]:
        df[f"{col}Group"] = grouping(df, group_column=col)
    df[f"DrivAgeGroup"] = grouping(df, group_column="DrivAge", 
                                   bins=[17, 25, 35, 45, 55, 65, df['DrivAge'].max()],
                                #    bins=[17, 25, 65, df['DrivAge'].max()],
                                   )

    bins   = [49, 50, 90, 99, 100, 230]
    labels = ["50", "51-90", "91-99", "100", "101+"]
    df["BonusMalusGroup"] = grouping(df, "BonusMalus", bins=bins, labels=labels)
    categorical_cols = [
    "DrivAgeGroup",
    "BonusMalusGroup",
    "VehPowerGroup",
    "VehAgeGroup",
    "logDensityGroup",
    "VehGas",
    "VehBrand",
    "Region",
    # "Area"
    ]
    train, test = one_hot_encode(df, categorical_cols)

    model_name = "v0.0.1"
    if model_path(model_name).exists():
        results = sm.load(model_path(model_name))
    else:
        results = poisson_glm(train, categorical_cols, model_name=model_name)

    mu_test = offset_test(results, train, test)
    save_dir = Path(config.SAVED_MU_DIR / 'poissonGLM' / model_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    test = test.assign(mu=mu_test)                     # 인덱스 기준으로 정확히 맞춰 붙음
    test[["IDpol", "ClaimNb", "Exposure", "mu"]].to_parquet(save_dir / "test_pred.parquet")
    noise = measure_noise(mu_test, n_sim=100, seed=0)
    print(f"Mean deviance: {noise.mean()}, Std deviance: {noise.std()}")
    print(f"Mean Poisson deviance on test set: {mean_poisson_deviance(test['ClaimNb'], mu_test)}")
    print(f"Baseline Poisson deviance on test set: {baseline_deviance(train, test)[0]}")
    print(results.summary())
