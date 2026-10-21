import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mip.config import Config
config = Config()

model_name = "v0.0.1"
test = pd.read_parquet(config.SAVED_MU_DIR / 'poissonGLM' / model_name / "test_pred.parquet")
test["predicted_frequency"] = test["mu"] / test["Exposure"]
test = test.sort_values("predicted_frequency")
cum_exp = test["Exposure"].cumsum() / test["Exposure"].sum()     # 0 → 1
test["decile"] = np.minimum((cum_exp * 10).astype(int), 9)         # 0 ~ 9
print(test)

cal = test.groupby("decile").agg(
    exposure=("Exposure", "sum"),
    pred_claims=("mu", "sum"),
    obs_claims=("ClaimNb", "sum"),
)
cal["pred_freq"] = cal["pred_claims"] / cal["exposure"]
cal["obs_freq"]  = cal["obs_claims"]  / cal["exposure"]
cal["obs_se"]    = np.sqrt(cal["obs_claims"]) / cal["exposure"]
print(cal)

x = cal.index
width = 0.4

plt.figure(figsize=(9, 5))
plt.bar(x - width/2, cal["pred_freq"], width, label="Predicted (GLM)")
plt.bar(x + width/2, cal["obs_freq"],  width, yerr=cal["obs_se"], capsize=3, label="Observed")

baseline_freq = test["ClaimNb"].sum() / test["Exposure"].sum()
plt.axhline(baseline_freq, color="gray", linestyle="--", label="Baseline")

plt.xlabel("Decile of predicted frequency (0 = safest, 9 = riskiest)")
plt.ylabel("Claims per policy-year")
plt.title("Calibration: predicted vs observed frequency")
plt.xticks(x)
plt.legend()
plt.savefig(config.FIG_DIR / "poissonGLM" / "calibration.png", bbox_inches="tight")

"""
4. 왜 사람 수가 아니라 Exposure로 같게 하나?

보험에서 "정보의 양"은 사람 수가 아니라 가입 기간이기 때문이에요.

1개월만 가입한 사람: 사고를 낼 기회가 1개월뿐이에요. 이 사람에게서 얻는 정보는 아주 적어요.
1년 가입한 사람: 1년 동안의 기록이 있어요. 정보가 12배 많아요.

사람 수로 같게 나누면, 단기 계약자가 몰린 그룹은 실제로는 데이터가 적은 그룹이 돼요. Exposure로 같게 나누면 모든 그룹이 **같은 “1년 수”**를 담고 있어서 정보량이 진짜로 같아져요.
"""