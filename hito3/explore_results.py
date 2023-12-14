from pathlib import Path

import pandas as pd
import dtale


ROOT_DIR = Path.cwd().parent
DATA_DIR = ROOT_DIR / "data"

DATA_PATH = DATA_DIR / "result.csv"

df = pd.read_csv(DATA_PATH)

dta = dtale.show(df)

dta.open_browser()

input("Press Enter to continue...")
