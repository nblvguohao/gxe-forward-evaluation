"""WP4 (plan v2 §10.4): numerical check of the weighted t* proposition (docs/wp4_tstar_proposition_2026-09-30.md).

Predictions  yhat_ij = m_j + t * d_ij  with d_ij a fixed within-environment profile (sum_i d_ij = 0 in every env).
Cell-weighted quantities (w_j = n_j / N):
  a   = sum_j w_j (m_j - mbar)(ybar_j - ybar)          between-env covariance of predicted and observed env means
  v_b = sum_j w_j (m_j - mbar)^2                        between-env variance of the predicted env means
  q   = sum_j w_j var_j(d)                              within-env variance of the profile (pooled)
  k   = sum_j w_j cov_j(d, y) / sqrt(q)                 pooled within-env covariance per unit profile SD
  S   = var(y)
Claim: pooled Pearson r(t) = (a + k u) / sqrt((v_b + u^2) S) with u = t sqrt(q); maximised at u* = k v_b / a (a > 0),
r_max = sqrt((a^2 / v_b + k^2) / S); pooled MSE is minimised at u = k whatever a. Exact identities, no distributional
assumptions: unequal environment sizes, unequal within-env variances and any dependence between m and d are allowed."""
import numpy as np

rng = np.random.default_rng(20260930)
worst = {"r_formula": 0.0, "u_star": 0.0, "r_max": 0.0, "u_mse": 0.0}
cases = 0
for rep in range(500):
    J = int(rng.integers(3, 40))
    n = rng.integers(3, 300, J)                      # unequal environment sizes
    env = np.repeat(np.arange(J), n)
    mu = rng.normal(0, rng.uniform(0.5, 5), J)       # environment effects
    sig = rng.uniform(0.3, 3, J)                     # unequal within-env SDs
    g = rng.normal(size=len(env))
    y = mu[env] + sig[env] * g
    m = 0.6 * mu + rng.normal(0, rng.uniform(0.1, 3), J)   # predicted env means, correlated with the truth
    d = rng.uniform(-0.2, 1.0) * g + rng.normal(size=len(env)) * rng.uniform(0.2, 2)
    d = d * rng.uniform(0.2, 3, J)[env]              # env-specific profile scales
    d = d - (np.bincount(env, d) / n)[env]           # within-env mean zero
    w = n / n.sum()
    ybar_j = np.bincount(env, y) / n
    mbar = (w * m).sum()
    a = (w * (m - mbar) * (ybar_j - (w * ybar_j).sum())).sum()
    v_b = (w * (m - mbar) ** 2).sum()
    q = (w * np.bincount(env, d * d) / n).sum()
    cov_j = np.bincount(env, d * (y - ybar_j[env])) / n
    k = (w * cov_j).sum() / np.sqrt(q)
    S = y.var()
    if a <= 0 or k <= 0:
        continue
    cases += 1
    ts = np.linspace(0, 6 * k * v_b / a / np.sqrt(q), 4001)[1:]
    r_num = np.array([np.corrcoef(m[env] + t * d, y)[0, 1] for t in ts])
    u = ts * np.sqrt(q)
    r_form = (a + k * u) / np.sqrt((v_b + u ** 2) * S)
    worst["r_formula"] = max(worst["r_formula"], np.abs(r_num - r_form).max())
    u_star = k * v_b / a
    u_num = u[np.argmax(r_num)]
    worst["u_star"] = max(worst["u_star"], abs(u_num - u_star) / u_star - (u[1] - u[0]) / u_star)
    worst["r_max"] = max(worst["r_max"], abs(r_num.max() - np.sqrt((a * a / v_b + k * k) / S)))
    mse = np.array([((m[env] + t * d - y) ** 2).mean() for t in ts])
    worst["u_mse"] = max(worst["u_mse"], abs(u[np.argmin(mse)] - k) - (u[1] - u[0]))
print("cases with a > 0 and k > 0:", cases)
print("max |r_numeric - r_formula|        :", f"{worst['r_formula']:.2e}")
print("max relative error of u* (beyond grid step):", f"{max(worst['u_star'], 0):.2e}")
print("max |max r_numeric - r_max|        :", f"{worst['r_max']:.2e}")
print("max |argmin MSE - k| (beyond grid step):", f"{max(worst['u_mse'], 0):.2e}")
