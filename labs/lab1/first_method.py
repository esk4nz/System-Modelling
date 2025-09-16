import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


def expo_sample(n, lam):
    if lam <= 0:
        raise ValueError("λ має бути додатним.")
    rng = np.random.default_rng()
    x = np.empty(n)
    for i in range(n):
        u = rng.random()
        while u == 0.0:
            u = rng.random()
        xi = -np.log(u) / lam
        x[i] = xi
    return x


def mean_var(x, ddof=0):
    n = len(x)
    s = 0.0
    for xi in x:
        s += xi
    m = s / n

    ssd = 0.0
    for xi in x:
        d = xi - m
        ssd += d * d
    var = ssd / (n - ddof)

    return m, var


def chi_square_test_exp_equal_prob(x, lam, k=20, alpha=0.05):
    """
    χ²-тест узгодженості для експоненційного розподілу з ВІДОМИМ λ.
    Рівноймовірні інтервали: p_i = 1/k. Останній інтервал [x_{k-1}, ∞).
    """
    n = len(x)
    if n == 0:
        raise ValueError("Порожня вибірка")

    # 1) Формуємо межі (без eps):
    edges = np.empty(k + 1, dtype=float)
    edges[0] = 0.0
    for i in range(1, k):  # i = 1 .. k-1
        q = i / k
        edges[i] = -np.log1p(-q) / lam   # стабільно для q близьких до 1
    edges[k] = np.inf  # останній бін

    # 2) Спостережені частоти
    observed, _ = np.histogram(x, bins=edges)

    # 3) Теоретичні ймовірності (усі 1/k) і очікування
    expected = np.full(k, n / k, dtype=float)

    # 4) Злиття бінів з малими очікуваннями (зазвичай не потрібно, бо n/k велике)
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

    # 5) Статистика χ²
    chi_stat = np.sum((obs_m - exp_m)**2 / exp_m)
    chi_crit = chi2.ppf(1 - alpha, df)
    p_value = chi2.sf(chi_stat, df)
    decision = "Не відкидаємо H0" if p_value >= alpha else "Відкидаємо H0"

    # (Перевірка суми)
    # print("sum expected =", exp_m.sum(), "sum observed =", obs_m.sum())

    return chi_stat, chi_crit, df, p_value, decision, k_eff



if __name__ == "__main__":
    n = 10_000
    lambdas = [0.5, 1.0, 2.0]
    BINS = 40
    SHOW_THEORETICAL = True
    X_PCTL = 0.99
    ALPHA = 0.05

    print(f"{'λ':>6} {'mean':>12} {'var':>12} {'mean_theor':>12} {'var_theor':>12} "
          f"{'rel_err_mean,%':>16} {'rel_err_var,%':>16} "
          f"{'chi2_stat':>12} {'chi2_crit':>12} {'df':>4} {'p-value':>10} {'decision':>15}")

    for lam in lambdas:
        x = expo_sample(n, lam)

        mean_sample, var_sample = mean_var(x, ddof=0)
        mean_theor = 1.0 / lam
        var_theor = 1.0 / (lam * lam)
        rel_err_mean = abs(mean_sample - mean_theor) / mean_theor
        rel_err_var = abs(var_sample - var_theor) / var_theor

        chi_stat, chi_crit, df, p_value, decision, k_eff = chi_square_test_exp_equal_prob(
            x, lam, k=BINS, alpha=ALPHA
        )

        print(
            f"{lam:6.2f} {mean_sample:12.6f} {var_sample:12.6f} "
            f"{mean_theor:12.6f} {var_theor:12.6f} "
            f"{100 * rel_err_mean:16.2f} {100 * rel_err_var:16.2f} "
            f"{chi_stat:12.3f} {chi_crit:12.3f} {df:4d} {p_value:10.3f} {decision:>15}"
        )

        plt.figure(figsize=(6, 4))
        plt.hist(
            x, bins=BINS, density=True, alpha=0.55,
            color="tab:blue", edgecolor="black", linewidth=0.4,
            label="Гістограма (нормована)"
        )

        if SHOW_THEORETICAL:
            x_max_plot = np.quantile(x, X_PCTL)
            xs = np.linspace(0, x_max_plot, 400)
            pdf = lam * np.exp(-lam * xs)
            plt.plot(xs, pdf, "r-", lw=2, label="Теоретична щільність λ e^{-λx}")
            plt.xlim(0, x_max_plot)

        plt.title(f"Експоненційний розподіл (λ={lam:g}, n={n}, df={df}, k_eff={k_eff})")
        plt.xlabel("x")
        plt.ylabel("Щільність")
        plt.legend()
        plt.tight_layout()

    plt.show()
