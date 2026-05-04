import os
import re
import json
import argparse
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--smooth',   type=int, default=10)
    parser.add_argument('--n_prev',   type=int, default=0,
                        help='Number of previous runs (by time) to overlay for comparison')
    return parser.parse_args()


_DT_PATTERN = re.compile(r'(\d{8}-\d{6})')
DUAL_LINESTYLES = {
    'eta':        '-',
    'kl_mu':      '--',
    'kl_sigma':   '-.',
    'alpha_mu':   ':',
    'alpha_sigma': (0, (3, 1, 1, 1, 1, 1)),
}


def run_datetime(name):
    m = _DT_PATTERN.search(name)
    return datetime.strptime(m.group(1), '%Y%m%d-%H%M%S') if m else None


def sorted_runs(logs_dir):
    """All runs with a performance.csv, sorted newest-first."""
    runs = []
    for d in os.listdir(logs_dir):
        if not os.path.isdir(os.path.join(logs_dir, d)):
            continue
        if not os.path.isfile(os.path.join(logs_dir, d, 'performance.csv')):
            continue
        dt = run_datetime(d)
        if dt:
            runs.append((dt, d))
    return [name for _, name in sorted(runs, reverse=True)]


def select_runs(logs_dir, run_name, n_prev):
    all_runs = sorted_runs(logs_dir)
    if not all_runs:
        return []
    primary = run_name or all_runs[0]
    try:
        idx = all_runs.index(primary)
    except ValueError:
        idx = 0
    selected = [primary] + all_runs[idx + 1: idx + 1 + n_prev]
    return selected


def parse_tensor_col(series):
    return series.apply(
        lambda x: float(str(x).replace('tensor(', '').replace(')', ''))
        if pd.notna(x) else float('nan')
    )


def load_run(logs_dir, run_name, smooth):
    run_dir = os.path.join(logs_dir, run_name)
    df = pd.read_csv(os.path.join(run_dir, 'performance.csv'))

    tensor_cols = [c for c in df.columns if df[c].astype(str).str.startswith('tensor').any()]
    for col in tensor_cols:
        df[col] = parse_tensor_col(df[col])

    df['_returns_smooth'] = df['mean_return'].rolling(smooth, min_periods=1).mean()

    with open(os.path.join(run_dir, 'hyperparams.json')) as f:
        params = json.load(f)
    training = params['training']
    critic_start = training['learning_starts']
    policy_offset = training.get('policy_learning_starts', training.get('policy_learning_start', 0))
    return df, critic_start, critic_start + policy_offset


def add_vlines(ax, critic_start, policy_start, color, legend=False):
    kw = dict(linestyle='--', linewidth=1.0, alpha=0.6)
    ax.axvline(critic_start, color='gray',  **kw, label='critic starts' if legend else '_')
    ax.axvline(policy_start, color=color,   **kw, label='policy starts' if legend else '_')


def main():
    args    = arg_parser()
    logs_dir = os.path.join(os.getcwd(), 'train_logs')

    runs = select_runs(logs_dir, args.run_name, args.n_prev)
    if not runs:
        print('No runs found.')
        return

    # colour palette: primary run is highlighted, comparison runs are muted
    palette = cm.tab10(np.linspace(0, 0.9, max(len(runs), 1)))

    fig = plt.figure(figsize=(12, 10))
    gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)
    ax_reward = fig.add_subplot(gs[0, :])
    ax_policy = fig.add_subplot(gs[1, 0])
    ax_critic = fig.add_subplot(gs[1, 1])
    ax_dual   = fig.add_subplot(gs[2, :])

    dual_cols_seen = set()

    for i, run_name in enumerate(runs):
        color  = palette[i]
        alpha  = 0.85 if i == 0 else 0.55
        lw     = 1.5  if i == 0 else 1.0
        label  = run_name

        try:
            df, critic_start, policy_start = load_run(logs_dir, run_name, args.smooth)
        except Exception as e:
            print(f'Skipping {run_name}: {e}')
            continue

        # --- reward ---
        ax_reward.plot(df['timestep'], df['_returns_smooth'],
                       color=color, alpha=alpha, lw=lw, label=label)
        add_vlines(ax_reward, critic_start, policy_start, color,
                   legend=(i == 0))

        # --- losses ---
        for ax, col in [(ax_policy, 'policy_loss'), (ax_critic, 'critic_loss')]:
            if col in df.columns:
                data = df[['timestep', col]].dropna()
                ax.plot(data['timestep'], data[col],
                        color=color, alpha=alpha, lw=lw, label=label)
            add_vlines(ax, critic_start, policy_start, color)

        # --- dual variables ---
        dual_cols = [c for c in ['eta', 'alpha_mu', 'alpha_sigma',
                                  'log_eta', 'log_alpha_mu', 'log_alpha_sigma']
                     if c in df.columns]
        for col in dual_cols:
            ls  = DUAL_LINESTYLES[col]
            lbl = f'{col} ({run_name})' if len(runs) > 1 else col
            ax_dual.plot(df['timestep'], df[col],
                         color=color, alpha=alpha, lw=lw, label=lbl, linestyle=ls)
            dual_cols_seen.add(col)

    # --- decorations ---
    primary_name = runs[0]
    ax_reward.set(xlabel='timestep', ylabel='mean return', xlim=(0, None),
                  title=f'Training return — {primary_name}' + (f' (+{len(runs)-1} prev)' if len(runs) > 1 else ''))
    ax_reward.grid(True, alpha=0.3)
    ax_reward.legend(fontsize='x-small', ncols=2)

    for ax, title in [(ax_policy, 'Policy loss'), (ax_critic, 'Critic loss')]:
        ax.set(xlabel='timestep', title=title, xlim=(0, None))
        ax.grid(True, alpha=0.3)

    ax_dual.set(xlabel='timestep', title='Dual variables', xlim=(0, None))
    ax_dual.grid(True, alpha=0.3)
    if dual_cols_seen:
        ax_dual.legend(fontsize='x-small', ncols=2)

    out_path = os.path.join(logs_dir, primary_name, 'dashboard.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved to {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
