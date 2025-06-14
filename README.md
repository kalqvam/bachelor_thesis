# ESG, Earnings, and Earthquakes: A Skeptical Analysis of Resilience in Macroeconomic Chaos

This repository contains the computational materials and analytical code developed for the bachelor's thesis "ESG, Earnings, and Earthquakes: A Skeptical Analysis of Resilience in Macroeconomic Chaos".

## Abstract

The thesis investigates the relationship between Environmental, Social, and Governance (ESG) performance and corporate resilience during macroeconomic shocks. Building upon stakeholder theory and employing a skeptical empirical approach, the study addresses critical methodological concerns often overlooked in ESG literature – namely, endogeneity, rater bias, and flow-driven performance concerns. Using a panel dataset of 577 U.S. firms from the Russell 3000 index and quarterly observations spanning 2017 to 2024, the analysis applies a System Generalised Method of Moments (GMM) framework to assess firm profitability, proxied by EBITDA margins, in response to the COVID-19 pandemic and the Russian invasion of Ukraine. ESG scores are normalised to capture relative performance, while firm-specific controls and interaction terms account for exogenous shocks. Results reveal limited effects: while ESG variables show no significant protective effect during the pandemic, environmental performance positively correlates with profitability following the invasion, whereas social performance has an adverse impact. Governance, contrary to prevailing literature, is found to be immaterial in both periods. These findings contrast with conventional narratives around ESG benefits, underscoring the need for a more rigorous approach to modelling techniques.

## Research Questions

1. Is relative ESG performance, measured through ESG scores, associated with better firm performance?
2. Does relative ESG performance provide protection to firms when exposed to external shocks?
3. Which ESG aspects have the most significant effect on firm performance, and in what direction?

## Key Findings

The results reveal limited protective effects of ESG during crises:
- **COVID-19 period (2018-2021)**: No statistically significant relationship between ESG scores and financial performance
- **Ukraine war period (2021-2024)**: Mixed results with environmental factors showing positive correlation with profitability, while social performance had adverse impacts
- **Governance**: Found to be immaterial in both periods, contrary to prevailing literature

## Methodology

### Theoretical Framework
- **Foundation**: Stakeholder theory
- **Approach**: Skeptical empirical analysis addressing methodological biases
- **Performance Metric**: EBITDA margins (accounting-based to avoid market flow concerns)
- **ESG Scoring**: Normalized relative scores to account for rater bias

### Analytical Approach
- **Primary Method**: System Generalised Method of Moments (GMM)
- **Time Periods**: Two separate analyses (2018-2021 and 2021-2024)
- **Sample**: 577 U.S. companies from Russell 3000 index
- **Data Source**: Financial Modelling Prep API, FMP ESG scores methodology is based on NLP techniques

## Repository Structure

```
esg_gmm/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
│
├── src/
│   ├── data_acquisition/
│   │   ├── api_client.py
│   │   ├── financial_data.py
│   │   ├── esg_data.py
│   │   ├── profile_data.py
│   │   └── etf_holdings.py
│   │
│   ├── preprocessing/
│   │   ├── interpolation.py
│   │   ├── dummyfication.py
│   │   │
│   │   ├── cleaning/
│   │   │   ├── column_management.py
│   │   │   ├── data_filtering.py
│   │   │   ├── date_processing.py
│   │   │   ├── duplicates.py
│   │   │   ├── missing_data.py
│   │   │   └── panel_validation.py
│   │   │
│   │   └── declowning/
│   │       ├── patterns.py
│   │       ├── outliers.py
│   │       ├── shit_filter.py
│   │       └── utils.py
│   │
│   ├── kalman/
│   │   ├── parameter_tuning.py
│   │   └── imputation.py
│   │
│   ├── transformations/
│   │   ├── calculations.py
│   │   └── seasonality.py
│   │
│   ├── analysis/
│   │   ├── diagnostics.py
│   │   ├── correlations.py
│   │   ├── variance_analysis.py
│   │   │
│   │   └── unit_root_tests/
│   │       ├── panic_core.py
│   │       ├── decomposition.py
│   │       ├── bootstrap.py
│   │       └── panel_diagnostics.py
│   │
│   └── utils/
│       ├── constants.py
│       ├── file_handlers.py
│       ├── logging_helpers.py
│       └── data_validation.py
│
└── r_scripts/
   ├── gmm_covid_shock.R
   └── gmm_war_shock.R
```

## Data Processing Pipeline

The analysis follows a 17-step workflow executed in Google Colab:

1. **Data Acquisition**: API requests to Financial Modelling Prep
2. **General Cleaning**: Initial data validation and formatting
3. **Missing Value Treatment**: Interpolation techniques
4. **Outlier Detection**: Pattern-based and statistical outlier removal ("declowning")
5. **ESG Data Validation**: Missing data assessment
6. **Kalman Filter Optimization**: Parameter tuning using Optuna
7. **ESG Imputation**: Application of optimized Kalman filters
8. **Feature Engineering**: Column management and calculations
9. **Seasonality Removal**: Detrending and seasonal adjustments
10. **Ratio Validation**: Removal of erroneous financial ratios
11. **Correlation Analysis**: Cross-variable relationship assessment
12. **Variance Testing**: ESG data variance validation
13. **Final Data Preparation**: Last preprocessing steps
14. **Unit Root Testing**: PANIC methodology for panel stationarity
15. **Model Diagnostics**: Validation procedures
16. **GMM Analysis**: System GMM estimation for both shock periods
17. **Results Compilation**: Final output generation

## Technical Implementation

### Key Libraries and Tools
- **Python**: Primary language for data processing and analysis
- **R**: GMM estimation using `linearmodels` and `plm` packages
- **Optuna**: Hyperparameter optimization for Kalman filters
- **Statsmodels**: Statistical analysis and diagnostics
- **Google Colab**: Development and execution environment

## Methodological Contributions

This study addresses three critical biases in ESG research:

1. **Rater Bias**: Using normalized relative ESG scores instead of raw ratings. Rater bias was demonstrated by Berg, Kölbel & Rigobon (2019)
2. **Flow-Driven Performance**: Employing accounting-based metrics (EBITDA) rather than market-based returns. Flow-driven performance was demonstrated by van der Beck (2024)
3. **Simultaneity Bias**: Implementing System GMM to address endogeneity concerns

## License and Attribution

This work is made available under a non-exclusive license for academic and educational purposes. While the code and methodology are shared openly, please cite appropriately if using this work in academic research.

**Citation**:
```
Haitin, M. (2025). ESG, Earnings, and Earthquakes: A Skeptical Analysis of Resilience in 
Macroeconomic Chaos. Bachelor's thesis, Tallinn University of Technology.
```

## Acknowledgments

The author acknowledges the substantial role of large language models (Claude by Anthropic and ChatGPT by OpenAI) in the technical implementation of this research, while emphasizing that all methodological decisions, interpretations, and conclusions remain the independent work of the author.

---

*This repository serves as a comprehensive record of the analytical process underlying the thesis, demonstrating the complexity and challenges inherent in modern empirical ESG research.*
