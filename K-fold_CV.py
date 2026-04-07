# Grouped company-level K-fold cross-validation for the HMM.
# Assumes the helper functions and constants from the main model cell are already defined.

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = print


# Cross-validation settings

CV_N_SPLITS = 5
CV_RANDOM_SEED = 123
CV_MAX_ITER = MAX_ITER
CV_TOL = TOL
CV_VERBOSE = True

import os

CV_OUTPUT_DIR = "/content/cv_outputs"
os.makedirs(CV_OUTPUT_DIR, exist_ok=True)

def cv_out(name):
    return os.path.join(CV_OUTPUT_DIR, name)

# Output files
CV_OUT_METRICS              = cv_out("cv_fold_metrics.csv")
CV_OUT_METRICS_SUMMARY      = cv_out("cv_metrics_summary.csv")
CV_OUT_ABS_BY_FOLD          = cv_out("cv_absorption_by_fold.csv")
CV_OUT_ABS_SUMMARY          = cv_out("cv_absorption_summary.csv")
CV_OUT_SIZE_BY_FOLD         = cv_out("cv_size_probs_by_fold.csv")
CV_OUT_SIZE_SUMMARY         = cv_out("cv_size_probs_summary.csv")
CV_OUT_TRANS_BY_FOLD        = cv_out("cv_marginal_transition_probs_by_fold.csv")
CV_OUT_TRANS_SUMMARY        = cv_out("cv_marginal_transition_probs_summary.csv")
CV_OUT_COEF_BY_FOLD         = cv_out("cv_transition_coefs_by_fold.csv")
CV_OUT_COEF_SUMMARY         = cv_out("cv_transition_coefs_summary.csv")
CV_OUT_EMISS_BY_FOLD        = cv_out("cv_emissions_by_fold.csv")
CV_OUT_EMISS_SUMMARY        = cv_out("cv_emissions_summary.csv")


# Data preparation helpers

def _load_cv_source_df():
    """Load the source dataframe for cross-validation."""
    required = ["company_id", "t", "obs_token", "log_raise", "delta_days"]

    if "df" in globals():
        _df = globals()["df"].copy()
        if all(c in _df.columns for c in required):
            _df = _df[required].copy()
            _df["obs_token"] = _df["obs_token"].astype(str)
            _df["delta_days"] = pd.to_numeric(_df["delta_days"], errors="coerce")
            _df = _df.sort_values(["company_id", "t"], kind="mergesort").reset_index(drop=True)
            return _df

    _df = pd.read_csv(INPUT_FILE)
    missing = [c for c in required if c not in _df.columns]
    if missing:
        raise ValueError(f"Input missing required columns for CV: {missing}")

    _df = _df[required].copy()
    _df["obs_token"] = _df["obs_token"].astype(str)
    _df["delta_days"] = pd.to_numeric(_df["delta_days"], errors="coerce")
    _df = _df.sort_values(["company_id", "t"], kind="mergesort").reset_index(drop=True)
    return _df


def _prepare_df_for_model(df_in, obs_vocab, r_mu=None, r_sd=None):
    """Prepare observation indices and standardized covariates."""
    out = df_in.copy()
    obs_index = {o: k for k, o in enumerate(obs_vocab)}

    unknown = sorted(set(out["obs_token"].astype(str).unique()) - set(obs_vocab))
    if unknown:
        raise ValueError(
            "Test fold contains observation tokens not seen in training fold: "
            f"{unknown}"
        )

    out["obs_idx"] = out["obs_token"].map(obs_index).astype(int)

    if r_mu is None or r_sd is None:
        r_std, r_mu, r_sd = standardize_feature(out)
    else:
        r = out["log_raise"].to_numpy()
        mask = np.isfinite(r)
        r_std = np.zeros_like(r, dtype=float)
        r_std[mask] = (r[mask] - r_mu) / (r_sd if r_sd > 1e-8 else 1.0)
        r_std[~mask] = 0.0

    miss = (~np.isfinite(out["log_raise"].to_numpy())).astype(float)
    out["r_std"] = r_std
    out["r_miss"] = miss

    out = out.sort_values(["company_id", "t"], kind="mergesort").reset_index(drop=True)
    return out, r_mu, r_sd


def _build_sequences(df_ready):
    sequences = []
    for cid, g in df_ready.groupby("company_id", sort=False):
        sequences.append((
            int(cid),
            g["obs_idx"].to_numpy(dtype=int),
            g["r_std"].to_numpy(dtype=float),
            g["r_miss"].to_numpy(dtype=float),
            g["delta_days"].to_numpy(dtype=float),
        ))
    return sequences


# Model fitting and scoring helpers

def fit_hmm_on_dataframe(df_train_raw, obs_vocab, max_iter=CV_MAX_ITER, tol=CV_TOL, verbose=CV_VERBOSE):
    """Fit the HMM on one training fold."""
    df_train, r_mu, r_sd = _prepare_df_for_model(df_train_raw, obs_vocab=obs_vocab)

    sequences = _build_sequences(df_train)

    n_states = len(STATES)
    n_obs = len(obs_vocab)

    B = initialize_emissions(obs_vocab)
    pi = initialize_pi()

    p = 3  # intercept, r_std, r_missing
    params = {}
    for i in range(n_states):
        m_i = len(ALLOWED_NEXT[i])
        params[i] = np.zeros((max(m_i - 1, 0), p), dtype=float)

    prev_ll = None
    xi_total = np.zeros((n_states, n_states), dtype=float)

    for it in range(1, max_iter + 1):
        pi_acc = np.zeros(n_states, dtype=float)
        emiss_counts = np.zeros((n_states, n_obs), dtype=float)
        xi_total = np.zeros((n_states, n_states), dtype=float)

        trans_X = {i: [] for i in range(n_states)}
        trans_W = {i: [] for i in range(n_states)}

        ll_total = 0.0

        for _, obs_seq, r_seq, m_seq, gap_seq in sequences:
            T = len(obs_seq)
            if T == 0:
                continue

            A_list = build_transition_matrices_for_sequence(r_seq, m_seq, params)
            logB = np.log(np.maximum(B[:, obs_seq], 1e-300))
            log_pi = np.log(np.maximum(pi, 1e-300))

            # Forward pass
            log_alpha = np.full((T, n_states), -np.inf)
            log_alpha[0, :] = log_pi + logB[:, 0]

            for t in range(1, T):
                logA = np.log(np.maximum(A_list[t - 1], 1e-300))
                tmp = log_alpha[t - 1, :][:, None] + logA
                log_alpha[t, :] = logB[:, t] + logsumexp(tmp, axis=0)

            loglik = float(logsumexp(log_alpha[T - 1, :], axis=0))
            ll_total += loglik

            # Backward pass
            log_beta = np.full((T, n_states), -np.inf)
            log_beta[T - 1, :] = 0.0

            for t in range(T - 2, -1, -1):
                logA = np.log(np.maximum(A_list[t], 1e-300))
                tmp = logA + (logB[:, t + 1] + log_beta[t + 1, :])[None, :]
                log_beta[t, :] = logsumexp(tmp, axis=1)

            # State posteriors
            log_gamma = log_alpha + log_beta - loglik
            gamma = np.exp(log_gamma)

            pi_acc += gamma[0, :]
            for t in range(T):
                emiss_counts[:, obs_seq[t]] += gamma[t, :]

            # Transition posteriors
            for t in range(1, T):
                logA = np.log(np.maximum(A_list[t - 1], 1e-300))
                log_xi = (
                    log_alpha[t - 1, :][:, None]
                    + logA
                    + logB[:, t][None, :]
                    + log_beta[t, :][None, :]
                    - loglik
                )
                xi = np.exp(log_xi)
                xi_total += xi

                x_t = np.array([1.0, float(r_seq[t]), float(m_seq[t])], dtype=float)
                for i in range(n_states):
                    choices = ALLOWED_NEXT[i]
                    if len(choices) <= 1:
                        continue
                    w = np.array([xi[i, j] for j in choices], dtype=float)
                    if w.sum() > 1e-12:
                        trans_X[i].append(x_t)
                        trans_W[i].append(w)

            # Terminal pseudo-transition
            gap_last = float(gap_seq[-1]) if T > 0 else np.nan
            q_fail = fixed_terminal_fail_prob(
                delta_days=gap_last,
                last_obs_idx=int(obs_seq[-1]),
                obs_vocab=obs_vocab
            )

            gamma_last = gamma[-1, :]
            xi_terminal = build_terminal_pseudo_xi(gamma_last, q_fail, ALLOWED_NEXT)
            xi_terminal *= TERMINAL_FAIL_WEIGHT

            xi_total += xi_terminal

            x_terminal = np.array([1.0, float(r_seq[-1]), float(m_seq[-1])], dtype=float)
            for i in range(n_states):
                choices = ALLOWED_NEXT[i]
                if len(choices) <= 1:
                    continue
                w = np.array([xi_terminal[i, j] for j in choices], dtype=float)
                if w.sum() > 1e-12:
                    trans_X[i].append(x_terminal)
                    trans_W[i].append(w)

        # M-step updates
        pi = np.maximum(pi_acc, 1e-300)
        pi = pi / pi.sum()

        B = emiss_counts + EMISSION_DIRICHLET
        B = B / B.sum(axis=1, keepdims=True)

        for i in range(n_states):
            if len(ALLOWED_NEXT[i]) <= 1:
                continue
            X_i = np.array(trans_X[i], dtype=float)
            W_i = np.array(trans_W[i], dtype=float)
            if X_i.shape[0] == 0:
                continue
            params[i] = fit_weighted_multinomial_logit(X_i, W_i, ridge_l2=RIDGE_L2)

        if verbose:
            print(f"    EM iter {it:02d} | train log-likelihood: {ll_total:,.3f}")

        if prev_ll is not None and abs(ll_total - prev_ll) < tol * (1.0 + abs(prev_ll)):
            if verbose:
                print("    Converged.")
            break
        prev_ll = ll_total

    counts = xi_total.copy()
    probs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-300)

    df_counts = pd.DataFrame(counts, index=STATES, columns=STATES)
    df_probs  = pd.DataFrame(probs,  index=STATES, columns=STATES)
    df_emiss  = pd.DataFrame(B, index=STATES, columns=obs_vocab)
    df_pi     = pd.DataFrame({"state": STATES, "pi": pi})
    df_abs    = compute_absorption_probabilities(probs, STATE_INDEX["Exit"], STATE_INDEX["Fail"])
    df_size   = transition_probs_at_size_table(params, r_mu=r_mu, r_sd=r_sd)

    return {
        "r_mu": r_mu,
        "r_sd": r_sd,
        "obs_vocab": obs_vocab,
        "B": B,
        "pi": pi,
        "params": params,
        "counts": counts,
        "probs": probs,
        "df_counts": df_counts,
        "df_probs": df_probs,
        "df_emiss": df_emiss,
        "df_pi": df_pi,
        "df_abs": df_abs,
        "df_size": df_size,
        "train_loglik": float(prev_ll if prev_ll is not None else np.nan),
        "n_train_companies": int(df_train["company_id"].nunique()),
        "n_train_events": int(len(df_train)),
    }


def score_hmm_on_dataframe(df_test_raw, fitted_model):
    """Score held-out observed-data log-likelihood on test companies."""
    df_test, _, _ = _prepare_df_for_model(
        df_test_raw,
        obs_vocab=fitted_model["obs_vocab"],
        r_mu=fitted_model["r_mu"],
        r_sd=fitted_model["r_sd"],
    )
    sequences = _build_sequences(df_test)

    B = fitted_model["B"]
    pi = fitted_model["pi"]
    params = fitted_model["params"]

    n_states = len(STATES)
    ll_total = 0.0
    n_companies = 0
    n_events = 0

    for _, obs_seq, r_seq, m_seq, _ in sequences:
        T = len(obs_seq)
        if T == 0:
            continue

        A_list = build_transition_matrices_for_sequence(r_seq, m_seq, params)
        logB = np.log(np.maximum(B[:, obs_seq], 1e-300))
        log_pi = np.log(np.maximum(pi, 1e-300))

        log_alpha = np.full((T, n_states), -np.inf)
        log_alpha[0, :] = log_pi + logB[:, 0]

        for t in range(1, T):
            logA = np.log(np.maximum(A_list[t - 1], 1e-300))
            tmp = log_alpha[t - 1, :][:, None] + logA
            log_alpha[t, :] = logB[:, t] + logsumexp(tmp, axis=0)

        ll_total += float(logsumexp(log_alpha[T - 1, :], axis=0))
        n_companies += 1
        n_events += T

    return {
        "test_loglik": float(ll_total),
        "n_test_companies": int(n_companies),
        "n_test_events": int(n_events),
        "test_loglik_per_company": float(ll_total / max(n_companies, 1)),
        "test_loglik_per_event": float(ll_total / max(n_events, 1)),
    }


# Output reshaping helpers

def extract_coef_rows(params):
    feature_names = ["intercept", "r_std", "r_missing"]
    rows = []
    for i in range(len(STATES)):
        choices = ALLOWED_NEXT[i]
        if len(choices) <= 1:
            continue
        ref = choices[0]
        Theta = params[i]
        for k in range(1, len(choices)):
            j = choices[k]
            for f in range(Theta.shape[1]):
                rows.append({
                    "from_state": STATES[i],
                    "to_state": STATES[j],
                    "reference_to_state": STATES[ref],
                    "feature": feature_names[f],
                    "coef": float(Theta[k - 1, f]),
                })
    return pd.DataFrame(rows)


def melt_transition_probs(df_probs):
    rows = []
    for from_state in STATES:
        for to_state in STATES:
            rows.append({
                "from_state": from_state,
                "to_state": to_state,
                "prob": float(df_probs.loc[from_state, to_state]),
            })
    return pd.DataFrame(rows)


def melt_emissions(df_emiss):
    rows = []
    for state in df_emiss.index.tolist():
        for obs_token in df_emiss.columns.tolist():
            rows.append({
                "state": state,
                "obs_token": obs_token,
                "prob": float(df_emiss.loc[state, obs_token]),
            })
    return pd.DataFrame(rows)


def summarize_with_dispersion(df_in, group_cols, value_col):
    out = (
        df_in.groupby(group_cols, dropna=False)[value_col]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={
            "mean": f"{value_col}_mean",
            "std": f"{value_col}_sd",
            "min": f"{value_col}_min",
            "max": f"{value_col}_max",
        })
    )
    out[f"{value_col}_sd"] = out[f"{value_col}_sd"].fillna(0.0)
    return out


def summarize_coef_stability(df_coef):
    grp = (
        df_coef.groupby(["from_state", "to_state", "reference_to_state", "feature"], dropna=False)["coef"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "coef_mean", "std": "coef_sd", "min": "coef_min", "max": "coef_max"})
    )
    grp["coef_sd"] = grp["coef_sd"].fillna(0.0)

    sign_stats = (
        df_coef.assign(
            positive=lambda x: (x["coef"] > 0).astype(float),
            negative=lambda x: (x["coef"] < 0).astype(float),
            nonzero=lambda x: (x["coef"] != 0).astype(float),
        )
        .groupby(["from_state", "to_state", "reference_to_state", "feature"], dropna=False)[["positive", "negative", "nonzero"]]
        .mean()
        .reset_index()
        .rename(columns={
            "positive": "share_positive",
            "negative": "share_negative",
            "nonzero": "share_nonzero",
        })
    )

    return grp.merge(sign_stats, on=["from_state", "to_state", "reference_to_state", "feature"], how="left")


# Fold construction

def make_grouped_company_folds(company_ids, n_splits=5, seed=123):
    company_ids = np.array(pd.Index(company_ids).drop_duplicates())
    if n_splits < 2:
        raise ValueError("CV_N_SPLITS must be at least 2.")
    if n_splits > len(company_ids):
        raise ValueError(f"CV_N_SPLITS={n_splits} exceeds number of companies={len(company_ids)}.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(company_ids)
    folds = [np.array(x) for x in np.array_split(shuffled, n_splits)]
    return folds


# Run cross-validation

cv_source_df = _load_cv_source_df()
global_obs_vocab = sorted(cv_source_df["obs_token"].astype(str).unique().tolist())

company_ids_all = cv_source_df["company_id"].drop_duplicates().tolist()
fold_company_ids = make_grouped_company_folds(
    company_ids=company_ids_all,
    n_splits=CV_N_SPLITS,
    seed=CV_RANDOM_SEED
)

metrics_rows = []
abs_by_fold = []
size_by_fold = []
trans_by_fold = []
coef_by_fold = []
emiss_by_fold = []

for fold_num, test_ids in enumerate(fold_company_ids, start=1):
    test_id_set = set(test_ids.tolist())
    train_mask = ~cv_source_df["company_id"].isin(test_id_set)
    test_mask = cv_source_df["company_id"].isin(test_id_set)

    df_train_fold = cv_source_df.loc[train_mask].copy()
    df_test_fold = cv_source_df.loc[test_mask].copy()

    if CV_VERBOSE:
        print(f"\n================ FOLD {fold_num}/{CV_N_SPLITS} ================")
        print(f"Train companies: {df_train_fold['company_id'].nunique():,} | Test companies: {df_test_fold['company_id'].nunique():,}")

    fitted = fit_hmm_on_dataframe(
        df_train_raw=df_train_fold,
        obs_vocab=global_obs_vocab,
        max_iter=CV_MAX_ITER,
        tol=CV_TOL,
        verbose=CV_VERBOSE
    )

    scored = score_hmm_on_dataframe(df_test_fold, fitted)

    metrics_rows.append({
        "fold": fold_num,
        "train_companies": int(df_train_fold["company_id"].nunique()),
        "test_companies": int(df_test_fold["company_id"].nunique()),
        "train_events": int(len(df_train_fold)),
        "test_events": int(len(df_test_fold)),
        "train_loglik": float(fitted["train_loglik"]),
        "train_loglik_per_company": float(fitted["train_loglik"] / max(df_train_fold["company_id"].nunique(), 1)),
        "train_loglik_per_event": float(fitted["train_loglik"] / max(len(df_train_fold), 1)),
        "test_loglik": float(scored["test_loglik"]),
        "test_loglik_per_company": float(scored["test_loglik_per_company"]),
        "test_loglik_per_event": float(scored["test_loglik_per_event"]),
    })

    tmp_abs = fitted["df_abs"].copy()
    tmp_abs["fold"] = fold_num
    abs_by_fold.append(tmp_abs)

    tmp_size = fitted["df_size"].copy()
    tmp_size["fold"] = fold_num
    size_by_fold.append(tmp_size)

    tmp_trans = melt_transition_probs(fitted["df_probs"])
    tmp_trans["fold"] = fold_num
    trans_by_fold.append(tmp_trans)

    tmp_coef = extract_coef_rows(fitted["params"])
    tmp_coef["fold"] = fold_num
    coef_by_fold.append(tmp_coef)

    tmp_emiss = melt_emissions(fitted["df_emiss"])
    tmp_emiss["fold"] = fold_num
    emiss_by_fold.append(tmp_emiss)


# Build summary outputs

cv_metrics = pd.DataFrame(metrics_rows)

cv_metrics_summary = pd.DataFrame({
    "metric": [
        "train_loglik",
        "train_loglik_per_company",
        "train_loglik_per_event",
        "test_loglik",
        "test_loglik_per_company",
        "test_loglik_per_event",
    ],
    "mean": [
        cv_metrics["train_loglik"].mean(),
        cv_metrics["train_loglik_per_company"].mean(),
        cv_metrics["train_loglik_per_event"].mean(),
        cv_metrics["test_loglik"].mean(),
        cv_metrics["test_loglik_per_company"].mean(),
        cv_metrics["test_loglik_per_event"].mean(),
    ],
    "sd": [
        cv_metrics["train_loglik"].std(ddof=1),
        cv_metrics["train_loglik_per_company"].std(ddof=1),
        cv_metrics["train_loglik_per_event"].std(ddof=1),
        cv_metrics["test_loglik"].std(ddof=1),
        cv_metrics["test_loglik_per_company"].std(ddof=1),
        cv_metrics["test_loglik_per_event"].std(ddof=1),
    ],
    "min": [
        cv_metrics["train_loglik"].min(),
        cv_metrics["train_loglik_per_company"].min(),
        cv_metrics["train_loglik_per_event"].min(),
        cv_metrics["test_loglik"].min(),
        cv_metrics["test_loglik_per_company"].min(),
        cv_metrics["test_loglik_per_event"].min(),
    ],
    "max": [
        cv_metrics["train_loglik"].max(),
        cv_metrics["train_loglik_per_company"].max(),
        cv_metrics["train_loglik_per_event"].max(),
        cv_metrics["test_loglik"].max(),
        cv_metrics["test_loglik_per_company"].max(),
        cv_metrics["test_loglik_per_event"].max(),
    ],
})
cv_metrics_summary["sd"] = cv_metrics_summary["sd"].fillna(0.0)

cv_abs_by_fold = pd.concat(abs_by_fold, ignore_index=True)
cv_abs_summary = (
    cv_abs_by_fold.groupby("state", dropna=False)
    .agg(
        prob_absorb_exit_mean=("prob_absorb_exit", "mean"),
        prob_absorb_exit_sd=("prob_absorb_exit", "std"),
        prob_absorb_exit_min=("prob_absorb_exit", "min"),
        prob_absorb_exit_max=("prob_absorb_exit", "max"),
        prob_absorb_fail_mean=("prob_absorb_fail", "mean"),
        prob_absorb_fail_sd=("prob_absorb_fail", "std"),
        prob_absorb_fail_min=("prob_absorb_fail", "min"),
        prob_absorb_fail_max=("prob_absorb_fail", "max"),
    )
    .reset_index()
)
for c in ["prob_absorb_exit_sd", "prob_absorb_fail_sd"]:
    cv_abs_summary[c] = cv_abs_summary[c].fillna(0.0)

cv_size_by_fold = pd.concat(size_by_fold, ignore_index=True)
cv_size_summary = summarize_with_dispersion(
    cv_size_by_fold,
    group_cols=["from_state", "deal_size_usd_mn", "to_state"],
    value_col="prob"
)

cv_trans_by_fold = pd.concat(trans_by_fold, ignore_index=True)
cv_trans_summary = summarize_with_dispersion(
    cv_trans_by_fold,
    group_cols=["from_state", "to_state"],
    value_col="prob"
)

cv_coef_by_fold = pd.concat(coef_by_fold, ignore_index=True)
cv_coef_summary = summarize_coef_stability(cv_coef_by_fold)

cv_emiss_by_fold = pd.concat(emiss_by_fold, ignore_index=True)
cv_emiss_summary = summarize_with_dispersion(
    cv_emiss_by_fold,
    group_cols=["state", "obs_token"],
    value_col="prob"
)

# Display summaries

print("\n\n==================== K-FOLD CV COMPLETE ====================\n")

print("Fold-by-fold held-out fit:")
display(cv_metrics.style.format({
    "train_loglik": "{:,.2f}",
    "train_loglik_per_company": "{:,.4f}",
    "train_loglik_per_event": "{:,.4f}",
    "test_loglik": "{:,.2f}",
    "test_loglik_per_company": "{:,.4f}",
    "test_loglik_per_event": "{:,.4f}",
}))

print("\nSummary of held-out fit metrics:")
display(cv_metrics_summary.style.format({"mean": "{:,.4f}", "sd": "{:,.4f}", "min": "{:,.4f}", "max": "{:,.4f}"}))

print("\nCross-fold absorption probability summary:")
display(cv_abs_summary.style.format({
    "prob_absorb_exit_mean": "{:.4f}",
    "prob_absorb_exit_sd": "{:.4f}",
    "prob_absorb_fail_mean": "{:.4f}",
    "prob_absorb_fail_sd": "{:.4f}",
}))

print("\nCross-fold size-conditioned transition summary (first 30 rows):")
display(cv_size_summary.head(30).style.format({
    "prob_mean": "{:.4f}",
    "prob_sd": "{:.4f}",
    "prob_min": "{:.4f}",
    "prob_max": "{:.4f}",
}))

print("\nCross-fold coefficient stability summary (r_std rows first):")
display(
    cv_coef_summary[cv_coef_summary["feature"] == "r_std"]
    .sort_values(["from_state", "to_state"])
    .style.format({
        "coef_mean": "{:.4f}",
        "coef_sd": "{:.4f}",
        "coef_min": "{:.4f}",
        "coef_max": "{:.4f}",
        "share_positive": "{:.2f}",
        "share_negative": "{:.2f}",
        "share_nonzero": "{:.2f}",
    })
)

# Export CSVs

cv_metrics.to_csv(CV_OUT_METRICS, index=False)
cv_metrics_summary.to_csv(CV_OUT_METRICS_SUMMARY, index=False)
cv_abs_by_fold.to_csv(CV_OUT_ABS_BY_FOLD, index=False)
cv_abs_summary.to_csv(CV_OUT_ABS_SUMMARY, index=False)
cv_size_by_fold.to_csv(CV_OUT_SIZE_BY_FOLD, index=False)
cv_size_summary.to_csv(CV_OUT_SIZE_SUMMARY, index=False)
cv_trans_by_fold.to_csv(CV_OUT_TRANS_BY_FOLD, index=False)
cv_trans_summary.to_csv(CV_OUT_TRANS_SUMMARY, index=False)
cv_coef_by_fold.to_csv(CV_OUT_COEF_BY_FOLD, index=False)
cv_coef_summary.to_csv(CV_OUT_COEF_SUMMARY, index=False)
cv_emiss_by_fold.to_csv(CV_OUT_EMISS_BY_FOLD, index=False)
cv_emiss_summary.to_csv(CV_OUT_EMISS_SUMMARY, index=False)

print("\nSaved CV CSVs to:", CV_OUTPUT_DIR)
print("\nFiles actually present there:")
for f in sorted(os.listdir(CV_OUTPUT_DIR)):
    print(" -", f)

# Zip and download outputs

import zipfile
zip_path = "/content/cv_outputs.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(os.listdir(CV_OUTPUT_DIR)):
        full_path = os.path.join(CV_OUTPUT_DIR, f)
        if os.path.isfile(full_path):
            zf.write(full_path, arcname=f)

print("\nCreated zip:", zip_path)

from google.colab import files
files.download(zip_path)
