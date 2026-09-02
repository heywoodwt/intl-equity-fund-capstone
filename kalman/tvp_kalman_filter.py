import numpy as np


class KalmanFilter:
    """Small Kalman filter/smoother for a random-walk TVP regression.

    This implements the subset of pykalman used by tvp_kalman.ipynb:
    EM updates for transition_covariance and observation_covariance,
    plus filter, smooth, and loglikelihood.
    """

    def __init__(
        self,
        n_dim_obs,
        n_dim_state,
        transition_matrices,
        observation_matrices,
        initial_state_mean,
        initial_state_covariance,
        transition_covariance=None,
        observation_covariance=None,
        em_vars=None,
    ):
        if n_dim_obs != 1:
            raise ValueError("This local KalmanFilter supports one-dimensional observations only.")

        self.n_dim_obs = n_dim_obs
        self.n_dim_state = n_dim_state
        self.transition_matrices = np.asarray(transition_matrices, dtype=float)
        self.observation_matrices = np.asarray(observation_matrices, dtype=float)
        self.initial_state_mean = np.asarray(initial_state_mean, dtype=float)
        self.initial_state_covariance = np.asarray(initial_state_covariance, dtype=float)
        self.transition_covariance = (
            np.eye(n_dim_state) * 1e-4
            if transition_covariance is None
            else np.asarray(transition_covariance, dtype=float)
        )
        self.observation_covariance = (
            np.array([[1e-4]])
            if observation_covariance is None
            else np.asarray(observation_covariance, dtype=float)
        )
        self.em_vars = set(em_vars or [])

    @staticmethod
    def _as_observations(observations):
        y = np.asarray(observations, dtype=float)
        if y.ndim == 2:
            y = y[:, 0]
        return y

    @staticmethod
    def _regularize_cov(cov, min_variance=1e-10):
        cov = (cov + cov.T) / 2
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, min_variance)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _filter_full(self, observations):
        y = self._as_observations(observations)
        n_obs = len(y)
        n_state = self.n_dim_state
        transition = self.transition_matrices
        q = self._regularize_cov(self.transition_covariance)
        r = float(np.asarray(self.observation_covariance)[0, 0])
        r = max(r, 1e-10)

        pred_means = np.zeros((n_obs, n_state))
        pred_covs = np.zeros((n_obs, n_state, n_state))
        filt_means = np.zeros_like(pred_means)
        filt_covs = np.zeros_like(pred_covs)
        loglik = 0.0

        for t in range(n_obs):
            if t == 0:
                pred_means[t] = self.initial_state_mean
                pred_covs[t] = self._regularize_cov(self.initial_state_covariance)
            else:
                pred_means[t] = transition @ filt_means[t - 1]
                pred_covs[t] = self._regularize_cov(transition @ filt_covs[t - 1] @ transition.T + q)

            h = self.observation_matrices[t, 0].reshape(1, -1)
            innovation = y[t] - float(h @ pred_means[t])
            innovation_var = float(h @ pred_covs[t] @ h.T + r)
            innovation_var = max(innovation_var, 1e-10)
            gain = (pred_covs[t] @ h.T / innovation_var).reshape(-1)

            filt_means[t] = pred_means[t] + gain * innovation
            filt_covs[t] = pred_covs[t] - np.outer(gain, gain) * innovation_var
            filt_covs[t] = self._regularize_cov(filt_covs[t])

            loglik += -0.5 * (np.log(2 * np.pi * innovation_var) + innovation**2 / innovation_var)

        return pred_means, pred_covs, filt_means, filt_covs, loglik

    def filter(self, observations):
        _, _, filt_means, filt_covs, _ = self._filter_full(observations)
        return filt_means, filt_covs

    def smooth(self, observations):
        pred_means, pred_covs, filt_means, filt_covs, _ = self._filter_full(observations)
        n_obs = len(filt_means)
        smoothed_means = filt_means.copy()
        smoothed_covs = filt_covs.copy()
        transition = self.transition_matrices

        for t in range(n_obs - 2, -1, -1):
            pred_cov_next = self._regularize_cov(pred_covs[t + 1])
            smoother_gain = filt_covs[t] @ transition.T @ np.linalg.pinv(pred_cov_next)
            smoothed_means[t] = filt_means[t] + smoother_gain @ (
                smoothed_means[t + 1] - pred_means[t + 1]
            )
            smoothed_covs[t] = filt_covs[t] + smoother_gain @ (
                smoothed_covs[t + 1] - pred_cov_next
            ) @ smoother_gain.T
            smoothed_covs[t] = self._regularize_cov(smoothed_covs[t])

        return smoothed_means, smoothed_covs

    def em(self, observations, n_iter=10):
        y = self._as_observations(observations)

        if {"transition_covariance", "observation_covariance"}.issubset(self.em_vars):
            try:
                from scipy.optimize import minimize
            except ImportError:
                minimize = None

            if minimize is not None:
                initial_q = np.maximum(np.diag(self.transition_covariance), 1e-10)
                initial_r = max(float(self.observation_covariance[0, 0]), 1e-10)
                initial_params = np.log(np.r_[initial_q, initial_r])

                def objective(log_params):
                    variances = np.exp(log_params)
                    self.transition_covariance = np.diag(variances[:-1])
                    self.observation_covariance = np.array([[variances[-1]]])
                    return -self.loglikelihood(y)

                result = minimize(
                    objective,
                    initial_params,
                    method="L-BFGS-B",
                    bounds=[(np.log(1e-10), np.log(1.0))] * len(initial_params),
                    options={"maxiter": max(20, n_iter * 10)},
                )
                variances = np.exp(result.x if result.success else initial_params)
                self.transition_covariance = np.diag(variances[:-1])
                self.observation_covariance = np.array([[variances[-1]]])
                return self

        for _ in range(n_iter):
            smoothed_means, smoothed_covs = self.smooth(y)

            if "observation_covariance" in self.em_vars:
                residual_vars = []
                for t, obs in enumerate(y):
                    h = self.observation_matrices[t, 0].reshape(1, -1)
                    residual = obs - float(h @ smoothed_means[t])
                    residual_vars.append(residual**2 + float(h @ smoothed_covs[t] @ h.T))
                self.observation_covariance = np.array([[max(np.mean(residual_vars), 1e-10)]])

            if "transition_covariance" in self.em_vars:
                diffs = np.diff(smoothed_means, axis=0)
                q = diffs.T @ diffs / max(len(diffs), 1)
                self.transition_covariance = self._regularize_cov(q)

        return self

    def loglikelihood(self, observations):
        return self._filter_full(observations)[-1]
