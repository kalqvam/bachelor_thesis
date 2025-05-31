#install.packages(c("plm", "tidyverse"))
library(plm)
library(tidyverse)
library(lubridate)
library(gmm)

data <- read.csv("raw_ratios.csv")

data$e_covid <- data$environmentalScore_minmax_norm * data$covid
data$s_covid <- data$socialScore_minmax_norm * data$covid
data$g_covid <- data$governanceScore_minmax_norm * data$covid

data$e_war <- data$environmentalScore_minmax_norm * data$war
data$s_war <- data$socialScore_minmax_norm * data$war
data$g_war <- data$governanceScore_minmax_norm * data$war

data$date <- ymd(data$date)

restricted_data <- data %>%
  filter((year(date) == 2017 & quarter(date) == 4) | 
           (year(date) >= 2019 & year(date) <= 2021))

panel_data <- pdata.frame(restricted_data, index = c("ticker", "date"))

cat("Dimensions of panel data:", dim(panel_data), "\n")

gmm_model <- pgmm(
  ebitda_to_revenue_ratio ~ lag(ebitda_to_revenue_ratio, 1) + 
    lag(environmentalScore_minmax_norm, 1) + 
    lag(socialScore_minmax_norm, 1) + 
    lag(governanceScore_minmax_norm, 1) +
    covid +
    lag(e_covid, 1) +
    lag(s_covid, 1) +
    lag(g_covid, 1) +
    lag(cashAndCashEquivalents_to_totalAssets_ratio, 1) +
    lag(totalDebt_to_totalAssets_ratio, 1) +
    lag(ln_totalAssets, 1) |
    lag(ebitda_to_revenue_ratio, 3:4) +
    lag(environmentalScore_minmax_norm, 1:2) +
    lag(socialScore_minmax_norm, 1:2) +
    lag(governanceScore_minmax_norm, 1:2) +
    lag(cashAndCashEquivalents_to_totalAssets_ratio, 2:3) +
    lag(totalDebt_to_totalAssets_ratio, 2:3) +
    lag(ln_totalAssets, 2:3),
  data = panel_data,
  effect = "individual",
  model = "twosteps",
  transformation = "ld",
  collapse = FALSE
)

summary(gmm_model)

sargan_test <- sargan(gmm_model)
print(sargan_test)

coef_plot <- data.frame(
  variable = names(coef(gmm_model)),
  estimate = coef(gmm_model),
  se = sqrt(diag(vcov(gmm_model)))
)
coef_plot$lower <- coef_plot$estimate - 1.96 * coef_plot$se
coef_plot$upper <- coef_plot$estimate + 1.96 * coef_plot$se

ggplot(coef_plot, aes(x = variable, y = estimate)) +
  geom_point() +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  coord_flip() +
  theme_minimal() +
  labs(title = "GMM Coefficient Estimates with 95% Confidence Intervals",
       x = "Variables",
       y = "Coefficient Estimate")
