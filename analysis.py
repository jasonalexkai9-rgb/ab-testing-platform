import pandas as pd
import numpy as np
from scipy import stats

def run_ab_test(df):
    group_a = df[df['group'] == 'A']['conversion']
    group_b = df[df['group'] == 'B']['conversion']

    mean_a = group_a.mean()
    mean_b = group_b.mean()

    t_stat, p_value = stats.ttest_ind(group_a, group_b)

    lift = ((mean_b - mean_a) / mean_a) * 100

    ci = stats.t.interval(0.95, len(group_b)-1, loc=mean_b, scale=stats.sem(group_b))

    if p_value < 0.05:
        significance = "Statistically Significant ✅"
        winner = "B" if mean_b > mean_a else "A"
    else:
        significance = "Not Significant Yet ❌"
        winner = "No clear winner"

    return {
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "p_value": round(p_value, 4),
        "lift": round(lift, 2),
        "significance": significance,
        "winner": winner,
        "ci_low": round(ci[0], 4),
        "ci_high": round(ci[1], 4)
    }

def get_sample_data():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'group': ['A'] * n + ['B'] * n,
        'conversion': list(np.random.binomial(1, 0.10, n)) + list(np.random.binomial(1, 0.12, n))
    })
    return df