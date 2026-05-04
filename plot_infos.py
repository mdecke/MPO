import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--smooth', type=int, default=10)
    return parser.parse_args()


def most_recent_run(logs_dir):
    import re
    from datetime import datetime
    pattern = re.compile(r'(\d{8}-\d{6})')
    best, best_dt = None, None
    for d in os.listdir(logs_dir):
        if not os.path.isdir(os.path.join(logs_dir, d)):
            continue
        if not os.path.isfile(os.path.join(logs_dir, d, 'performance.csv')):
            continue
        m = pattern.search(d)
        if not m:
            continue
        dt = datetime.strptime(m.group(1), '%Y%m%d-%H%M%S')
        if best_dt is None or dt > best_dt:
            best, best_dt = d, dt
    return best


def parse_tensor_col(series):
    return series.apply(
        lambda x: float(str(x).replace('tensor(', '').replace(')', ''))
        if pd.notna(x) else float('nan')
    )


def add_vlines(ax, critic_start, policy_start, legend=False):
    kw = dict(linestyle='--', linewidth=1.0)
    ax.axvline(critic_start, color='gray',      **kw, label='critic learning starts' if legend else '_')
    ax.axvline(policy_start, color='steelblue', **kw, label='policy learning starts'  if legend else '_')


def main():
    args = arg_parser()
    logs_dir = os.path.join(os.getcwd(), 'train_logs')
    run_name = args.run_name or most_recent_run(logs_dir)
    if run_name is None:
        print('No runs found.')
        return

    run_dir = os.path.join(logs_dir, run_name)
    df = pd.read_csv(os.path.join(run_dir, 'performance.csv'))

    with open(os.path.join(run_dir, 'hyperparams.json')) as f:
        params = json.load(f)

    training = params['training']
    critic_start = training['learning_starts']
    policy_offset = training.get('policy_learning_starts', training.get('policy_learning_start', 0))
    policy_start  = critic_start + policy_offset

    tensor_cols = [c for c in df.columns if df[c].astype(str).str.startswith('tensor').any()]
    for col in tensor_cols:
        df[col] = parse_tensor_col(df[col])

    returns = df['mean_return'].rolling(args.smooth, min_periods=1).mean()

    dual_cols = [c for c in ['eta', 'alpha_mu', 'alpha_sigma', 'log_eta', 'log_alpha_mu', 'log_alpha_sigma']
                 if c in df.columns]

    fig = plt.figure(figsize=(12, 10))
    gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)

    ax_reward = fig.add_subplot(gs[0, :])
    ax_policy = fig.add_subplot(gs[1, 0])
    ax_critic = fig.add_subplot(gs[1, 1])
    ax_dual   = fig.add_subplot(gs[2, :])

    # --- reward ---
    ax_reward.plot(df['timestep'], returns, color='steelblue', alpha=0.85)
    ax_reward.set(xlabel='timestep', ylabel='mean return', xlim=(0, None),
                  title=f'Training return — {run_name}')
    ax_reward.grid(True, alpha=0.3)
    add_vlines(ax_reward, critic_start, policy_start, legend=True)
    ax_reward.legend(fontsize='small')

    # --- losses ---
    for ax, col, title in [
        (ax_policy, 'policy_loss', 'Policy loss'),
        (ax_critic, 'critic_loss', 'Critic loss'),
    ]:
        if col in df.columns:
            data = df[['timestep', col]].dropna()
            ax.plot(data['timestep'], data[col], alpha=0.75)
        ax.set(xlabel='timestep', title=title, xlim=(0, None))
        ax.grid(True, alpha=0.3)
        add_vlines(ax, critic_start, policy_start)

    # --- dual variables ---
    for col in dual_cols:
        ax_dual.plot(df['timestep'], df[col], label=col, alpha=0.75)
    ax_dual.set(xlabel='timestep', title='Dual variables', xlim=(0, None))
    ax_dual.grid(True, alpha=0.3)
    if dual_cols:
        ax_dual.legend(fontsize='small')
    add_vlines(ax_dual, critic_start, policy_start)

    out_path = os.path.join(run_dir, 'dashboard.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved to {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
