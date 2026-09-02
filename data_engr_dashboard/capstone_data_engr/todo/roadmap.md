# Roadmap

## Data Acquisition & Cleaning (Completed)

## Data Engineering & Enrichment: All we have are the holdings and trades of the fund. We need more informative data.

### Program data pipelines that enrich our dataset with additional information (e.g. sector for each holding, certain fundamentals) (In Progress)
- Obtain and clean the returns of the equities and the fund (Completed)
- Obtain and clean the baseline returns (Completed)
- Gather FF World Factors to include in each timepoint in the dataset. (Completed)
- Macro regime indicators (or monthly proxies): VIX, Inflation: PC & CPI, Money supply, Credit index, Yield curve indicators, US index (S&P 500)
- Gather and aggregate the fundamentals (valuation ratios, quality ratios, momentum, etc)
- Gather and data engineer category breakdowns (percentage in each sector, etc)
- Country information (dummyified identifiers, currency info, money supply, etc)

### Combine into a single dataset.
### Perform feature engineering through something like PCA to ensure linear independence.
### Post: After this step, we have a single dataframe with date, profile holding, FF World Factors, various fundamentals, and other information on the aggregate holdings of the fund helpful to deconstructing performance.

## Baseline Analysis: Do the simplest analysis and prediction steps first.
- Implement a rolling regression and plot the coefficients over time. Analyse them to gather insights.
- Expand the rolling regression to include non-linear basis functions and repeat the analysis. Include regularization to make the regressions and analysis tractable. (This is the thing he was telling us to do in the email)
- Post: After this step, we will have a good idea of how various factors influenced the performance of this fund over different regimes, and we will have some theories about what drove the performance on a more detailed level than linear regression can provide.

## Predictive Modeling: Predict alpha.
- Exploit the insights gathered in the last step to create deep learning models (attention, CNNs, RNNs, etc) to predict alpha. Formulate architectures so that we can mine insights out of the results.
- Post: After this step, we will have either verified or rejected our theories about what drove performance, and we will have one or more predictive models to help extend the project.

## Decide From There