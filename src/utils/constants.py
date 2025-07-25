DEFAULT_TICKER_COLUMN = 'ticker'
DEFAULT_DATE_COLUMNS = ['date', 'quarter', 'quarter_year', 'date_quarter']
DEFAULT_TIME_COLUMN = 'quarter'

FINANCIAL_COLUMNS = [
    'revenue', 'ebitda', 'totalAssets', 'totalDebt', 'cashAndCashEquivalents',
    'totalCurrentLiabilities', 'totalCurrentAssets'
]

ESG_COLUMNS = [
    'environmentalScore', 'socialScore', 'governanceScore', 'ESGScore'
]

ESG_RISK_COLUMNS = ['ESGRiskRating']

PROFILE_COLUMNS = ['sector', 'ipoDate', 'companyAge']

COMMON_RATIO_COLUMNS = [
    'ebitda_to_revenue_ratio',
    'totalDebt_to_totalAssets_ratio',
    'cashAndCashEquivalents_to_totalAssets_ratio'
]

NORMALIZED_COLUMNS = [
    'environmentalScore_minmax_norm',
    'socialScore_minmax_norm', 
    'governanceScore_minmax_norm'
]

LOG_TRANSFORMED_COLUMNS = ['ln_totalAssets']

SHOCK_DUMMY_COLUMNS = ['covid', 'war']

DEFAULT_ANALYSIS_COLUMNS = FINANCIAL_COLUMNS + ESG_COLUMNS + COMMON_RATIO_COLUMNS

DEFAULT_MISSING_THRESHOLD = 0.2
DEFAULT_CONSECUTIVE_MISSING_THRESHOLD = 9
DEFAULT_VARIANCE_THRESHOLD = 1e-5
DEFAULT_OUTLIER_THRESHOLD = 3
DEFAULT_MIN_OBSERVATIONS = 10
DEFAULT_CORRELATION_SIGNIFICANCE = 0.05
DEFAULT_SAMPLE_DISPLAY_SIZE = 10

API_RATE_LIMIT = 300
API_BATCH_SIZE = 50
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 3

QUARTER_YEAR_PATTERN = r'Q(\d)-(\d{4})'

CSV_DATETIME_FORMAT = '%Y-%m-%d'
OUTPUT_TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'
DISPLAY_TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

DEFAULT_OUTPUT_DIR = '.'
DEFAULT_RESULTS_DIR = 'results'
DEFAULT_KALMAN_DIR = 'kalman_results'
DEFAULT_COMPONENTS_DIR = 'kalman_components'

FILE_ENCODING = 'utf-8'
CSV_INDEX = False

PANIC_DEFAULT_BOOTSTRAP_SAMPLES = 449
PANIC_DEFAULT_CUMULATIVE_VAR_THRESHOLD = 0.95
PANIC_MAX_FACTORS = 25

KALMAN_DEFAULT_TRIALS = 30
KALMAN_MIN_CONSECUTIVE_OBS = 30
KALMAN_SIGMA_RANGE = (0.001, 1.0)

SEASONALITY_MAX_LAGS = 16
SEASONALITY_ALPHA = 0.05
SEASONALITY_MIN_DATA_POINTS = 8

EXCLUDED_COVID_PERIODS = [(2020, 1), (2020, 2)]
EXCLUDED_WAR_PERIODS = [(2022, 1), (2022, 2)]
EXCLUDED_PERIODS = EXCLUDED_COVID_PERIODS + EXCLUDED_WAR_PERIODS

COVID_SHOCK_YEARS = [2020]
COVID_SHOCK_QUARTERS = [1, 2]
WAR_SHOCK_YEARS = [2022, 2023, 2024]
WAR_SHOCK_QUARTERS = [1, 2, 3, 4]

CONSOLE_WIDTH = 80
PROGRESS_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'

NUMERIC_PRECISION = 4
PERCENTAGE_PRECISION = 2

CONSOLE_SEPARATORS = {
    'major': '=' * CONSOLE_WIDTH,
    'minor': '-' * CONSOLE_WIDTH,
    'section': '#' * 30
}
