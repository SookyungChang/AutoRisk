from sklearn.datasets import fetch_openml
from mip.config import Config
config = Config()

data_dir = config.DATA_DIR
data_dir.mkdir(parents=True, exist_ok=True)

freq = fetch_openml(data_id=41214, as_frame=True)
freq_df = freq.frame
freq_df.to_parquet(data_dir / "frequency.parquet", index=False)

sev = fetch_openml(data_id=41215, as_frame=True)
sev_df = sev.frame
sev_df.to_parquet(data_dir / "severity.parquet", index=False)