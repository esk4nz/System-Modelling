import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def lcg_sample(n, a=5**13, c=2**31):
    x = np.empty(n)
    z = 1
    for i in range(n):
        z = (a * z) % c
        x[i] = z / c
    return x

def mean_var(x, ddof=0):
    n = len(x)
    m = np.sum(x) / n
    var = np.sum((x - m)**2) / (n - ddof)
    return m, var

def chi_square_test_uniform(x, k=20, alpha=0.05):
    n = len(x)
    edges = np.linspace(0, 1, k + 1)
    observed, _ = np.histogram(x, bins=edges)
    expected = np.full(k, n / k, dtype=float)

    obs_m, exp_m = [], []
    acc_o = acc_e = 0.0
    for o, e in zip(observed, expected):
        acc_o += o
        acc_e += e
        if acc_e >= 5.0:
            obs_m.append(acc_o)
            exp_m.append(acc_e)
            acc_o = acc_e = 0.0
    if acc_e > 0.0:
        if obs_m:
            obs_m[-1] += acc_o
            exp_m[-1] += acc_e
        else:
            obs_m.append(acc_o)
            exp_m.append(acc_e)

    obs_m = np.array(obs_m, dtype=float)
    exp_m = np.array(exp_m, dtype=float)
    k_eff = len(exp_m)
    df = k_eff - 1
    if df <= 0:
        raise ValueError(f"Надто мало інтервалів після злиття, df={df}")

    chi_stat = np.sum((obs_m - exp_m)**2 / exp_m)
    chi_crit = chi2.ppf(1 - alpha, df)
    decision = "Не відкидаємо H0" if chi_stat < chi_crit else "Відкидаємо H0"

    return chi_stat, chi_crit, df, decision, k_eff

if __name__ == "__main__":
    n = 1000
    params = [
        {"a": 5**13, "c": 2**31},
        {"a": 7**5, "c": 2**31-1},
        {"a": 3**10, "c": 2**30}
    ]
    BINS = 40
    SHOW_THEORETICAL = True
    ALPHA = 0.05

    print(f"{'a':>12} {'c':>12} {'mean':>12} {'var':>12} "
          f"{'rel_err_mean,%':>16} {'rel_err_var,%':>16} "
          f"{'chi2_stat':>12} {'chi2_crit':>12} {'df':>4} {'decision':>15}")

    for p in params:
        x = lcg_sample(n, a=p["a"], c=p["c"])

        mean_sample, var_sample = mean_var(x, ddof=0)
        mean_theor = 0.5
        var_theor = 1/12
        rel_err_mean = abs(mean_sample - mean_theor) / mean_theor
        rel_err_var = abs(var_sample - var_theor) / var_theor

        chi_stat, chi_crit, df, decision, k_eff = chi_square_test_uniform(
            x, k=BINS, alpha=ALPHA
        )

        print(
            f"{p['a']:12d} {p['c']:12d} {mean_sample:12.6f} {var_sample:12.6f} "
            f"{100 * rel_err_mean:16.2f} {100 * rel_err_var:16.2f} "
            f"{chi_stat:12.3f} {chi_crit:12.3f} {df:4d} {decision:>15}"
        )

        plt.figure(figsize=(6, 4))
        plt.hist(
            x, bins=BINS, density=False, alpha=0.55,
            color="tab:orange", edgecolor="black", linewidth=0.4
        )
        plt.title(f"Гістограма частот (a={p['a']}, c={p['c']}, n={n})")
        plt.xlabel("x")
        plt.ylabel("Кількість")
        plt.tight_layout()

        plt.figure(figsize=(6, 4))
        plt.hist(
            x, bins=BINS, density=True, alpha=0.55,
            color="tab:blue", edgecolor="black", linewidth=0.4,
            label="Гістограма (нормована)"
        )

        if SHOW_THEORETICAL:
            xs = np.linspace(0, 1, 400)
            pdf = np.ones_like(xs)  # рівномірний pdf на [0,1)
            plt.plot(xs, pdf, "r-", lw=2, label="Теоретична щільність U(0,1)")
            plt.xlim(0, 1)

        plt.title(f"Рівномірний LCG (a={p['a']}, c={p['c']}, n={n})")
        plt.xlabel("x")
        plt.ylabel("Щільність")
        plt.legend()
        plt.tight_layout()

    plt.show()
