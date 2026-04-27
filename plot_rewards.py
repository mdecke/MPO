import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("csv", nargs="?", default="train_logs_layernorm.csv")
parser.add_argument("--smooth", type=int, default=10, help="rolling window size for smoothing")
args = parser.parse_args()

cwd = os.getcwd()
args.csv = os.path.join(cwd,args.csv)
df = pd.read_csv(args.csv)

returns = df["mean_return"]
if args.smooth > 1:
    returns = returns.rolling(args.smooth, min_periods=1).mean()

plt.figure(figsize=(10, 5))
plt.plot(df["timestep"], returns)
plt.xlabel("Environment interactions")
plt.ylabel("Mean return")
plt.title("Training return")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
