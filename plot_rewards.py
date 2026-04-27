import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--logs_dir", default="train_logs")
parser.add_argument("--smooth", type=int, default=10, help="rolling window size for smoothing")
args = parser.parse_args()

logs_dir = os.path.join(os.getcwd(), args.logs_dir)
csv_files = sorted(f for f in os.listdir(logs_dir) if f.endswith(".csv"))

if not csv_files:
    print(f"No CSV files found in {logs_dir}")
    exit(1)

plt.figure(figsize=(10, 5))

for fname in csv_files:
    df = pd.read_csv(os.path.join(logs_dir, fname))
    returns = df["mean_return"]
    if args.smooth > 1:
        returns = returns.rolling(args.smooth, min_periods=1).mean()
    label = os.path.splitext(fname)[0]
    plt.plot(df["timestep"], returns, label=label)

plt.xlabel("Environment interactions")
plt.ylabel("Mean return")
plt.title("Training return")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
