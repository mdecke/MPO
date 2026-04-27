import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--logs_dir", default="train_logs")
parser.add_argument("--smooth", type=int, default=10, help="rolling window size for smoothing")
args = parser.parse_args()

logs_dir = os.path.join(os.getcwd(), args.logs_dir)
run_dirs = sorted(
    d for d in os.listdir(logs_dir)
    if os.path.isdir(os.path.join(logs_dir, d))
    and os.path.isfile(os.path.join(logs_dir, d, "performance.csv"))
)

if not run_dirs:
    print(f"No run folders with performance.csv found in {logs_dir}")
    exit(1)

plt.figure(figsize=(10, 5))

for run in run_dirs:
    df = pd.read_csv(os.path.join(logs_dir, run, "performance.csv"))
    returns = df["mean_return"]
    if args.smooth > 1:
        returns = returns.rolling(args.smooth, min_periods=1).mean()
    plt.plot(df["timestep"], returns, label=run)

plt.xlabel("Environment interactions")
plt.ylabel("Mean return")
plt.title("Training return")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
