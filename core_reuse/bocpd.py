from __future__ import annotations
import numpy as np, math


def _logsumexp(a: np.ndarray) -> float:
    m = np.max(a)
    return m + math.log(np.exp(a - m).sum())


def _student_t_logpdf(x, mu, var, nu):
    var = np.maximum(var, 1e-12)
    nu = np.maximum(nu, 1e-6)
    z = (x - mu)**2 / var
    lg = np.vectorize(math.lgamma)
    return (
        lg((nu + 1.0) / 2.0) - lg(nu / 2.0)
        - 0.5 * (math.log(math.pi) + np.log(nu)) - 0.5 * np.log(var)
        - ((nu + 1.0) / 2.0) * np.log1p(z / nu)
    )


class BOCPD:
    """
    Bayesian Online Changepoint Detection (Adams & MacKay, 2007),
    Normal-Inverse-Gamma prior → Student-t predictive.
    - Constant hazard H = 1/λ
    - Run-length posterior R_t over r=0..rmax
    """
    def __init__(self, hazard_lambda=200, rmax=600,
                 mu0=0.0, kappa0=1e-3, alpha0=1.0, beta0=1.0):
        self.rmax = int(rmax)
        self.H = 1.0 / float(hazard_lambda)

        self.logR = np.full(self.rmax + 1, -np.inf)
        self.logR[0] = 0.0  # start fully at r=0

        self.mu   = np.full(self.rmax + 1, mu0, dtype=float)
        self.kappa= np.full(self.rmax + 1, kappa0, dtype=float)
        self.alpha= np.full(self.rmax + 1, alpha0, dtype=float)
        self.beta = np.full(self.rmax + 1, beta0, dtype=float)

        self._mu0, self._k0, self._a0, self._b0 = mu0, kappa0, alpha0, beta0

    def _pred_loglik(self, x: float) -> np.ndarray:
        mu, k, a, b = self.mu, self.kappa, self.alpha, self.beta
        k = np.maximum(k, 1e-12); a = np.maximum(a, 1e-12); b = np.maximum(b, 1e-12)
        nu = 2.0 * a
        var = b * (k + 1.0) / (a * k)
        x_arr = np.full_like(mu, float(x))
        return _student_t_logpdf(x_arr, mu, var, nu)

    def _posterior_update_params(self, mu, k, a, b, x):
        k1 = k + 1.0
        mu1 = (k * mu + x) / k1
        a1 = a + 0.5
        b1 = b + 0.5 * (k * (x - mu)**2) / k1
        return mu1, k1, a1, b1

    def update(self, x: float) -> float:
        log_pred = self._pred_loglik(x)
        log_growth = np.full_like(self.logR, -np.inf)
        log_growth[1:] = (np.log(1.0 - self.H) + log_pred[:-1] + self.logR[:-1])
        log_cp0 = math.log(self.H) + _logsumexp(log_pred + self.logR)

        logR_new = log_growth
        logR_new[0] = log_cp0
        logZ = _logsumexp(logR_new)
        self.logR = logR_new - logZ

        mu_new  = np.empty_like(self.mu)
        kap_new = np.empty_like(self.kappa)
        alp_new = np.empty_like(self.alpha)
        bet_new = np.empty_like(self.beta)

        mu_new[0], kap_new[0], alp_new[0], bet_new[0] = self._posterior_update_params(
            self._mu0, self._k0, self._a0, self._b0, x
        )
        mu_g, kg, ag, bg = self.mu[:-1], self.kappa[:-1], self.alpha[:-1], self.beta[:-1]
        mu_g1, kg1, ag1, bg1 = self._posterior_update_params(mu_g, kg, ag, bg, x)
        mu_new[1:], kap_new[1:], alp_new[1:], bet_new[1:] = mu_g1, kg1, ag1, bg1

        self.mu, self.kappa, self.alpha, self.beta = mu_new, kap_new, alp_new, bet_new
        return float(np.exp(self.logR[0]))  # cp_prob = P(run-length=0)
