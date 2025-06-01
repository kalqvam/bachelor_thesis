import pandas as pd
import numpy as np
import warnings
import os
from tqdm import tqdm
import optuna
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.stats.diagnostic import acorr_ljungbox
from datetime import datetime

warnings.filterwarnings('ignore')

def prepare_dataset(file_path, min_consecutive_observations=50):
    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")

    if 'quarter' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['quarter']):
            df['date_quarter'] = pd.to_datetime(df['quarter'])
        else:
            df['date_quarter'] = df['quarter']
    else:
        raise ValueError("Expected 'quarter' column not found in the dataset")

    df['year'] = df['date_quarter'].dt.year
    df.sort_values(['ticker', 'date_quarter'], inplace=True)

    esg_columns = ['environmentalScore', 'socialScore', 'governanceScore']

    valid_tickers = []
    print(f"Finding tickers with at least {min_consecutive_observations} consecutive non-missing observations...")

    for ticker in tqdm(df['ticker'].unique()):
        ticker_data = df[df['ticker'] == ticker].sort_values('date_quarter')

        has_enough_data = True
        for col in esg_columns:
            non_missing = ~ticker_data[col].isna()

            consecutive_counts = []
            count = 0
            for val in non_missing:
                if val:
                    count += 1
                else:
                    if count > 0:
                        consecutive_counts.append(count)
                    count = 0
            if count > 0:
                consecutive_counts.append(count)

            if not consecutive_counts or max(consecutive_counts) < min_consecutive_observations:
                has_enough_data = False
                break

        if has_enough_data:
            valid_tickers.append(ticker)

    print(f"Found {len(valid_tickers)} tickers with sufficient data.")

    df_filtered = df[df['ticker'].isin(valid_tickers)].copy()
    print(f"Filtered dataset shape: {df_filtered.shape}")

    return df_filtered, valid_tickers

def fit_kalman_model(data, ticker, column, sigma2_level, sigma2_obs, get_innovations=False):
    clean_data = data.dropna()

    try:
        model = UnobservedComponents(
            clean_data,
            level='local level'
        )

        initial_params = [sigma2_level, sigma2_obs]

        result = model.fit(initial_params, method='powell', maxiter=100, disp=False)

        filter_results = result.filter_results
        innovations = filter_results.forecasts_error[0]

        if get_innovations:
            return result, innovations
        else:
            return result, result.aic

    except Exception as e:
        print(f"Error fitting UC model for {ticker}, {column}: {e}")
        return None, None

def evaluate_white_noise(innovations, ticker, column, print_pvalues=True):
    lb_test = acorr_ljungbox(innovations, lags=[1, 2, 3, 4], return_df=True)
    p_values = lb_test['lb_pvalue'].values

    if print_pvalues:
        print(f"\nLjung-Box test p-values for {ticker}, {column}:")
        for lag, p_value in enumerate(p_values, 1):
            print(f"  Lag {lag}: p-value = {p_value:.4f} {'(passed)' if p_value > 0.05 else '(failed)'}")

    is_white_noise = all(p > 0.05 for p in p_values)
    lb_stat = lb_test['lb_stat'].iloc[-1]

    return is_white_noise, lb_stat, p_values

def calculate_harvey_likelihood_univariate(series, model_result):
    filter_results = model_result.filter_results

    innovations = filter_results.forecasts_error[0]

    forecast_error_variances = filter_results.forecasts_error_cov[0, 0, :]

    n = len(innovations)

    if n == 0:
        return -np.inf

    term1 = -n / 2 * np.log(2 * np.pi)

    term2 = -0.5 * np.sum(np.log(forecast_error_variances))

    term3 = -0.5 * np.sum(innovations**2 / forecast_error_variances)

    log_likelihood = term1 + term2 + term3

    return log_likelihood

def optimize_global_parameters(df, valid_tickers, target_column, optuna_trials=30, output_dir="kalman_results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nOptimizing global parameters for {target_column} using Harvey's univariate log-likelihood approach")

    def objective(trial):
        sigma2_level = trial.suggest_float("sigma2_level", 0.001, 1.0, log=True)
        sigma2_obs = trial.suggest_float("sigma2_obs", 0.001, 1.0, log=True)

        all_likelihoods = []
        successful_fits = 0
        total_points = 0

        for ticker in valid_tickers:
            ticker_data = df[df['ticker'] == ticker].sort_values('date_quarter')
            series = ticker_data[target_column]

            if series.dropna().size < 10:
                continue

            model_result, aic = fit_kalman_model(
                series, ticker, target_column, sigma2_level, sigma2_obs
            )

            if model_result is not None:
                log_likelihood = calculate_harvey_likelihood_univariate(series, model_result)
                series_length = series.dropna().size

                all_likelihoods.append(log_likelihood)
                total_points += series_length
                successful_fits += 1

        if not all_likelihoods or successful_fits < 0.5 * len(valid_tickers):
            return -float('inf')

        total_log_likelihood = sum(all_likelihoods)

        if trial.number % 5 == 0:
            print(f"\nTrial {trial.number} - parameters: σ²_lvl={sigma2_level:.6f}, σ²_obs={sigma2_obs:.6f}")
            print(f"  Total log-likelihood: {total_log_likelihood:.4f}")
            print(f"  Average log-likelihood per ticker: {total_log_likelihood/successful_fits:.4f}")
            print(f"  Successful fits: {successful_fits}/{len(valid_tickers)}")

        return total_log_likelihood

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=optuna_trials)

    if not np.isneginf(study.best_value):
        best_params = study.best_params
        best_sigma2_level = best_params['sigma2_level']
        best_sigma2_obs = best_params['sigma2_obs']

        print(f"\nBest global parameters for {target_column} (based on Harvey's univariate log-likelihood):")
        print(f"  sigma2_level: {best_sigma2_level:.6f}")
        print(f"  sigma2_obs: {best_sigma2_obs:.6f}")
        print(f"  Total log-likelihood: {study.best_value:.4f}")

        print("\nEvaluating white noise properties of the optimal model (Ljung-Box test):")
        white_noise_count = 0
        total_evaluated = 0
        all_pvalues = []

        for ticker in tqdm(valid_tickers, desc="Testing white noise"):
            ticker_data = df[df['ticker'] == ticker].sort_values('date_quarter')
            series = ticker_data[target_column]

            if series.dropna().size < 10:
                continue

            model_result, innovations = fit_kalman_model(
                series, ticker, target_column,
                best_sigma2_level, best_sigma2_obs,
                get_innovations=True
            )

            if model_result is not None and innovations is not None:
                is_white_noise, _, p_values = evaluate_white_noise(
                    innovations, ticker, target_column,
                    print_pvalues=(total_evaluated < 5)
                )

                if p_values is not None:
                    all_pvalues.extend(p_values)

                if is_white_noise:
                    white_noise_count += 1
                total_evaluated += 1

        white_noise_pct = (white_noise_count / total_evaluated * 100) if total_evaluated > 0 else 0
        print(f"Percentage of companies with white noise residuals: {white_noise_pct:.2f}% ({white_noise_count}/{total_evaluated})")

        if all_pvalues:
            avg_pvalue = np.mean(all_pvalues)
            print(f"Average p-value across all tests: {avg_pvalue:.4f}")

        return {
            'column': target_column,
            'sigma2_level': best_sigma2_level,
            'sigma2_obs': best_sigma2_obs,
            'log_likelihood': study.best_value,
            'white_noise_pct': white_noise_pct
        }
    else:
        print(f"No valid parameters found for {target_column}")
        return None

def extract_components_with_global_params(df, valid_tickers, target_column, global_params, output_dir="kalman_components"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nExtracting components for {target_column}")

    if not global_params:
        print(f"No parameters found for {target_column}, skipping")
        return None

    sigma2_level = global_params['sigma2_level']
    sigma2_obs = global_params['sigma2_obs']

    components = []

    for ticker in tqdm(valid_tickers, desc="Processing tickers"):
        ticker_data = df[df['ticker'] == ticker].sort_values('date_quarter')
        series = ticker_data[target_column]

        if series.dropna().size < 10:
            continue

        model_result, innovations = fit_kalman_model(
            series, ticker, target_column, sigma2_level, sigma2_obs, get_innovations=True
        )

        if model_result is not None:
            is_white_noise, _, _ = evaluate_white_noise(
                innovations, ticker, target_column, print_pvalues=False
            )

            states = model_result.smoother_results.smoothed_state

            data_indices = series.dropna().index

            for idx, date in zip(data_indices, series.dropna().index):
                i = data_indices.get_loc(idx)
                if i < len(states[0]):
                    trend = states[0][i]
                    original = series.loc[idx]
                    residual = original - trend

                    date_value = ticker_data.loc[idx, 'date_quarter']

                    components.append({
                        'ticker': ticker,
                        'date_quarter': date_value,
                        'column': target_column,
                        'original': original,
                        'trend': trend,
                        'residual': residual,
                        'is_white_noise': is_white_noise
                    })

    components_df = pd.DataFrame(components)

    if not components_df.empty:
        output_file = f"{output_dir}/kalman_components_{target_column}_{timestamp}.csv"
        components_df.to_csv(output_file, index=False)
        print(f"Components saved to {output_file}")
        return components_df
    else:
        print("No components extracted")
        return None

def run_kalman_for_all_scores(file_path="panel_data.csv",
                              min_consecutive_observations=50,
                              optuna_trials=30,
                              max_companies=None):
    print("Starting Kalman filter parameter optimization for all ESG scores using Harvey's univariate likelihood approach...")

    df, valid_tickers = prepare_dataset(file_path, min_consecutive_observations)

    if len(valid_tickers) == 0:
        print("No valid tickers found with sufficient data. Try lowering min_consecutive_observations.")
        return None, None

    if max_companies is not None and isinstance(max_companies, (int, float)):
        if max_companies > 0 and max_companies < len(valid_tickers):
            print(f"Limiting analysis to {max_companies} companies out of {len(valid_tickers)} valid tickers")
            valid_tickers = valid_tickers[:max_companies]

    esg_columns = ['environmentalScore', 'socialScore', 'governanceScore']

    optimal_params = []
    all_components_dfs = []

    for target_column in esg_columns:
        print(f"\n{'=' * 40}")
        print(f"Processing {target_column}")
        print(f"{'=' * 40}")

        global_params = optimize_global_parameters(
            df, valid_tickers, target_column,
            optuna_trials=optuna_trials
        )

        if global_params:
            optimal_params.append(global_params)

            components_df = extract_components_with_global_params(
                df, valid_tickers, target_column, global_params
            )

            if components_df is not None:
                all_components_dfs.append(components_df)
        else:
            print(f"No valid global parameters found for {target_column}. Component extraction skipped.")

    if optimal_params:
        optimal_params_df = pd.DataFrame(optimal_params)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "kalman_results"
        os.makedirs(output_dir, exist_ok=True)

        optimal_params_df.to_csv(f"{output_dir}/optimal_params_all_scores_{timestamp}.csv", index=False)
        print(f"\nOptimal parameters for all scores saved to {output_dir}/optimal_params_all_scores_{timestamp}.csv")

        print("\nSummary of optimal parameters for all scores:")
        print(optimal_params_df)
    else:
        optimal_params_df = None
        print("\nNo valid parameters found for any score.")

    if all_components_dfs:
        all_components = pd.concat(all_components_dfs, ignore_index=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "kalman_components"
        os.makedirs(output_dir, exist_ok=True)

        all_components.to_csv(f"{output_dir}/kalman_all_components_{timestamp}.csv", index=False)
        print(f"\nAll components for all scores saved to {output_dir}/kalman_all_components_{timestamp}.csv")

    print("\nKalman filter parameter optimization for all ESG scores completed!")

    return optimal_params_df, all_components_dfs
