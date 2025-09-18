import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

def normal_sample(n, a=0.0, sigma=1.0):
    rng = np.random.default_rng()
    x = np.empty(n)
    for i in range(n):
        mu_i = np.sum(rng.random(12)) - 6  # сума 12 ξ_j - 6
        x[i] = sigma * mu_i + a
    return x

def mean_var(x, ddof=0):
    n = len(x)
    m = np.sum(x) / n
    var = np.sum((x - m)**2) / (n - ddof)
    return m, var

def chi_square_test_norm(x, a, sigma, k=20, alpha=0.05):
    n = len(x)
    if n == 0:
        raise ValueError("Порожня вибірка")

    q = np.linspace(0, 1, k + 1)
    edges = norm.ppf(q, loc=a, scale=sigma)
    edges[0] = -np.inf
    edges[-1] = np.inf

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
    params = [(0.0, 1.0), (2.0, 0.5), (-1.0, 2.0)]
    BINS = 40
    SHOW_THEORETICAL = True
    X_PCTL = 0.99
    ALPHA = 0.05

    print(f"{'a':>6} {'sigma':>6} {'mean':>12} {'var':>12} {'mean_theor':>12} {'var_theor':>12} "
          f"{'rel_err_mean,%':>16} {'rel_err_var,%':>16} "
          f"{'chi2_stat':>12} {'chi2_crit':>12} {'df':>4} {'decision':>15}")

    for a, sigma in params:
        x = normal_sample(n, a, sigma)

        mean_sample, var_sample = mean_var(x, ddof=0)
        mean_theor = a
        var_theor = sigma**2
        rel_err_mean = abs(mean_sample - mean_theor) / mean_theor if mean_theor != 0 else abs(mean_sample)
        rel_err_var = abs(var_sample - var_theor) / var_theor

        chi_stat, chi_crit, df, decision, k_eff = chi_square_test_norm(
            x, a, sigma, k=BINS, alpha=ALPHA
        )

        print(
            f"{a:6.2f} {sigma:6.2f} {mean_sample:12.6f} {var_sample:12.6f} "
            f"{mean_theor:12.6f} {var_theor:12.6f} "
            f"{100 * rel_err_mean:16.2f} {100 * rel_err_var:16.2f} "
            f"{chi_stat:12.3f} {chi_crit:12.3f} {df:4d} {decision:>15}"
        )

        plt.figure(figsize=(6, 4))
        plt.hist(
            x, bins=BINS, density=False, alpha=0.55,
            color="tab:orange", edgecolor="black", linewidth=0.4
        )
        plt.title(f"Гістограма частот (a={a}, σ={sigma}, n={n})")
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
            x_max_plot = np.quantile(x, X_PCTL)
            xs = np.linspace(np.min(x), x_max_plot, 400)
            pdf = norm.pdf(xs, loc=a, scale=sigma)
            plt.plot(xs, pdf, "r-", lw=2, label="Теоретична щільність N(a,sigma^2)")
            plt.xlim(np.min(x), x_max_plot)

        plt.title(f"Нормальний розподіл (a={a}, σ={sigma}, n={n})")
        plt.xlabel("x")
        plt.ylabel("Щільність")
        plt.legend()
        plt.tight_layout()

    plt.show()
