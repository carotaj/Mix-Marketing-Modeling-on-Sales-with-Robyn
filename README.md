# Mix-Marketing-Modeling-on-Sales-with-Robyn

Author: Marcello Carota

The aim of this project is to present and use an analytical tool called Marketing Mix Modeling (MMM), an econometric model designed to measure the real impact of all factors influencing sales, prepare historical time series data of marketing activity and create a statistical model capable of adjusting investment to improve results. 

## Introduction 
The technology industry has grown exponentially over the years, revolutionizing, among others, the world of advertising, turning it into a hybrid, complex and diverse environment. 
Marketing specialists strive to demonstrate how marketing campaigns drive business results, but despite numerous technological advances, marketing measurement is becoming increasingly complex, since in addition to traditional advertising media such as newspapers, radio, television, etc., online media such as social networks, search engines, websites, etc., have also been added. The advantage of these digital media is that they are easy to track and analyze, but when we talk about traditional or offline media, the tracking of advertising activity is much more difficult due to its massive and unidirectional communication model, making it impossible to track the impacted users. 

## What is MMM?
Marketing mix modeling (MMM) is a privacy-friendly, highly resilient, data-driven statistical analysis that quantifies the incremental sales impact and ROI of marketing and non-marketing activities.
MMM is an econometric model that aims to quantify the incremental impact of marketing and non-marketing activities on a pre-defined KPI (like sales or website visits). This is a holistic model used to understand how to allocate a marketing budget across marketing channels, products and regions and can help forecast the impact of future events or campaigns.
It enables dynamic and actionable marketing decision making by modernizing MMM with machine learning techniques.

## How does MMM work?
MMM relies on regression modeling and it derives an equation that describes the KPI. This equation shows what a change in each variable means for the KPI. A series of independent variables that are expected to impact sales are used to predict the dependent variable.
When running an MMM study, there are multiple steps involved.

**1) Define business questions and scope**
**2) Data collection**
   - There are multiple factors that need to be considered when it comes to data collection:
     -- *Actual vs Panel Data*
     -- *Dependent vs Independent Variables*
     -- *Typology of metrics*
     -- *Paid vs Organic activity*
     -- *Seasonality*
     -- *Macroeconomic factors*
     -- *Granularity*
**3) Data review**
**4) Feature Engineering**
**5) Modeling**
**6) Analysis and recommendations**

Some common questions that are best answered by MMM include:

- *Which media channels (online and offline) generate more revenue?*

- *What was the ROI of each marketing channel?*

- *If I had to cut my marketing budget, which channel should I cut?*

- *How much do incremental revenues drive from commercial activities?*

## Key Features of Robyn:

The dataset used in this project consists of weekly observations with 21 variables, capturing sales data, marketing spends, economic indicators, and other relevant factors.

## Data Summary

**_Observations:_** _157 weeks (from 2019-06-30 to 2022-06-26)_

## Exploratory Data Analysis
A comprehensive Exploratory Data Analysis (EDA) was conducted to understand the data, identify patterns, and inform modeling decisions.

1) Data Quality Checks
2) Missing Values: No missing values were detected.
3) Outliers: Boxplots revealed outliers in some variables, which were investigated and addressed appropriately.
4) Data Types: Ensured correct data types (e.g., dates as Date, categorical variables as factor).

# Descriptive Statistics:

- Calculated summary statistics.

# Time Series Visualization:

Prophet has automatically been included in the Robyn code to decompose the data into trend, seasonality, holiday and weekday impacts, in order to improve the model fit and ability to forecast. Traditionally it would be required to collect and model seasonality and holiday data as separate dummy variables in the model.

# Correlation Analysis:

Found strong positive correlations between sales and certain marketing spends, especially tv and facebook newsfeed.

- Performed Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) analysis.

- Stationarity Tests

- Conducted Augmented Dickey-Fuller tests to assess stationarity.
 (Found that differencing was required to achieve stationarity for sales)

# Forecasting Models
To forecast future sales and understand the impact of different variables, several models were developed.

# 1) ARIMA
*Autoregressive Integrated Moving Average (ARIMA) models*,
are used for analyzing and forecasting time series data by capturing temporal structures.

Differenced the series to achieve stationarity.

Used ACF and PACF plots to determine the order of AR (p) and MA (q) components.

**Model Estimation:**
Utilized the auto.arima() function for automated model selection.

**Model Diagnostics:**

Checked residuals for autocorrelation using the Ljung-Box test.
Ensured residuals were normally distributed and homoscedastic.

**Forecasting:**
Generated forecasts for the next 12 weeks.
Visualized forecasts along with confidence intervals.

**Results:**
The ARIMA model captured the underlying patterns in sales.
Forecasts closely aligned with actual values in the validation period.

**Limitations: **
ARIMA models do not incorporate exogenous variables, limiting insights into external factors.

# 2) ARIMAX
_ARIMA with Exogenous Variables (ARIMAX)_ extends ARIMA by including external predictors.


**Variable Selection:**

Selected relevant exogenous variables (e.g., tv_spend, search_spend).
Checked for multicollinearity using Variance Inflation Factor (VIF).

Fitted the ARIMAX model using auto.arima() with the xreg parameter.
Ensured that exogenous variables were stationary or transformed appropriately.

**Model Diagnostics:**

Analyzed residuals to confirm no remaining autocorrelation.
Evaluated the significance of exogenous variables.

**Forecasting:**

Forecasted sales using future values of exogenous variables (if available).
Assessed the impact of marketing spends on future sales.

**Results:**
ARIMAX provided better forecasts by incorporating marketing activities.
Highlighted the influence of marketing channels on sales.

# 3) Multiple Regression
Multiple linear regression models quantify the relationship between a dependent variable and multiple independent variables.

Fitted the model using the lm() function.

**Results:**
The multiple regression model identified key drivers of sales.
Achieved a satisfactory Adjusted R-squared value.

**Limitations:**
Linear models may not capture complex nonlinear relationships.

## Robyn Model Implementation
Robyn was used to build an advanced MMM, leveraging its automated hyperparameter optimization and ability to model adstock and saturation effects.

# Data Preparation

**Specifying Variables:**

_Dependent Variable:_  **sales**
_Paid Media Spends:_ Included variables like **facebook_newsfeed_spend**, **tv_spend**, etc.
_Paid Media Variables:_ Corresponding metrics like **facebook_newsfeed_impressions**, **tv_gross_rating_points**.
_Context Variables:_ Economic indicators.
_Prophet Variables:_ Enabled **trend, season, and holiday** components.
_Hyperparameters_: Hyperparameters control the adstock and saturation transformations in the Robyn model.

# Setting Hyperparameters:
**Adstock Decay Rate (thetas):**

Determines how quickly the effect of media spend decays over time.
Set based on media channel characteristics.
