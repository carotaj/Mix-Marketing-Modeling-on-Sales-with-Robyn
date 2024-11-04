#############################################################################################
####################         Meta MMM Open Source: Robyn 3.10.3       #######################
####################               JetBrains Assignment               #######################
#############################################################################################
library(readr)
library(dplyr)
library(caTools)
library(randomForest)
library(forecast)
library(tseries)
library(readxl)
library(Robyn)
library(ggplot2)
library(reticulate)


#Lad Dataset
data <- read.csv("Desktop/ROBYN PROJECT/jiali/data_week_final.csv")
######################     EXPLORATORY DATA ANALYSIS     #######################   
######################                                   #######################

View(data)
dim(data)
str(data)
summary(data)
colSums(is.na(data))
table(duplicated(data))

data$week <- as.Date(data$week, format='%Y-%m-%d')
data$year <- substr(data$week, 1, 4)
data$year <- as.factor(data$year)

#plot sales over time
ggplot(data, aes(x = week, y = sales)) +
  geom_smooth(method = "auto", se = FALSE) +
  labs(title = "Sales Over Time",
       x = "Date",
       y = "Amount") 

#scatter plot of advertising channels versus sales
df<-data[,-1]
plot_data <- data[, c("sales", colnames(df)[-which(colnames(df) == "sales")])]
scatter_plots <- list()
for (col in colnames(plot_data)[-1]) {
  scatter_plots[[col]] <- ggplot(plot_data, aes_string(x = col, y = "sales")) +
    geom_point()
}
grid.arrange(grobs = scatter_plots, ncol = 3)

#boxplot for sales by year
ggplot(data, aes(x = factor(format(week, "%Y")), y = sales)) +
  geom_boxplot() +
  xlab("Year") +
  ylab("Sales") +
  ggtitle("Sales Distribution by Year")

#histogram and qq plot
hist <- ggplot(data, aes(x = sales)) +
  geom_histogram() +
  xlab("Sales") +
  ylab("Frequency") +
  ggtitle("Histogram of Sales")

qq <- ggplot(data, aes(sample = sales)) +
  geom_qq() +
  xlab("Theoretical Quantiles") +
  ylab("Sample Quantiles") +
  ggtitle("QQ Plot of Sales")

grid.arrange(hist, qq)

#heatmap of correlation matrix
cor_vars1 <- c("sales","facebook_newsfeed_spend","youtube_brand_spend",
               "search_spend","youtube_performance_spend", 
               "newspaper_spend","tv_spend")

cor_data1 <- data[cor_vars1]

# fill all missings with 0 to allow for correlation calculation
cor_data1[is.na(cor_data1) == TRUE] <- 0
corr_matrix <- cor(cor_data1)
corrplot(corr_matrix, method = "color", tl.col="black", tl.cex = 0.8)


####################        TIME SERIES DECOMPOSITION        ###################

ts_data <- ts(data$sales, start = c(2019,7,7), end = c(2022, 6,19), frequency = 52)
decomp_data <- decompose(ts_data, "multiplicative")
autoplot(decomp_data) +
  ggtitle("Time Series Decomposition") +
  theme(plot.title = element_text(hjust = 0.5))

#Time Series decomposition for a 2-year time window
ts_data_2 <- ts(data$sales, start = c(2020,1,5), end = c(2022,1,2), frequency = 52)
decomp_data_2 <- decompose(ts_data_2)
autoplot(decomp_data_2) +
  ggtitle("Time Series Decomposition 2020-2022") +
  theme(plot.title = element_text(hjust = 0.5))

# ACF plot
Acf(ts_data, main = "ACF of Sales")

# PACF plot
Pacf(ts_data, main = "PACF of Sales")

#Augmented Dickey-Fuller Test
adf.test(data$sales, alternative = "stationary") #p-value is not smaller than 0.05 indicating that the time series is non-stationary


############################   FORECASTING MODELS   ############################   

########################           Arima Model          ########################

#step 1 : Differnece the Series
sales_ts_diff <- diff(ts_data, differences = 1)

# Plot differenced series
plot(sales_ts_diff, main = "Differenced Sales Time Series", ylab = "Differenced Sales", xlab = "Time")

# ADF test on differenced series : #p-value smaller than 0.05
adf.test(sales_ts_diff, alternative = "stationary") 

#step 2 : Identify ARIMA parameters (p,d,q)
# ACF and PACF of differenced series
Acf(sales_ts_diff, main = "ACF of Differenced Sales")
Pacf(sales_ts_diff, main = "PACF of Differenced Sales")

#step 3 : Fit ARIMA Model
# Fit ARIMA model
train_size <- floor(0.8 * nrow(data))
train_ts_data <- ts_data[1:train_size]
test_ts_data <- ts_data[126:length(ts_data)]

arima_model <- auto.arima(train_ts_data, max.q = 8, max.p = 2, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
summary(arima_model)

# Check residuals
checkresiduals(arima_model)
#Lijung-Box Test : the p value is greater than 0.05 then the residuals are independent
Box.test(arima_model$residuals, lag = 20, type = "Ljung-Box") 

#forecasting the next 12 weeks
forecast_horizon <- length(ts_data)-length(train_ts_data)
sales_forecast <- forecast(train_ts_data, h = forecast_horizon)

#Plot Predicted vs Actual Sales for Arima Model
results_arima <- data.frame(
  Actual = test_ts_data,
  Predicted = sales_forecast$mean
)

results_arima$Date <- data$week[126:156]

ggplot(results_arima, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual Sales")) +
  geom_line(aes(y = Predicted, color = "Predicted Sales"), linetype = "dashed") +
  labs(
    title = "Actual vs. Predicted Sales Over Time",
    x = "Date",
    y = "Sales"
  ) +
  scale_color_manual("", values = c("Actual Sales" = "blue", "Predicted Sales" = "red")) +
  theme_minimal()



##############    ArimaX : Incorporating Exogenous Variables      ##############

# Prepare exogenous variables (make sure they are aligned and have no missing values)
exogenous_vars <- data.frame(
  unemployment = data$unemployment,
  temperature = data$temperature,
  fb_spend = data$facebook_newsfeed_spend,
  fb_impressions = data$facebook_newsfeed_impressions,
  yt_brand_spend = data$youtube_brand_spend,
  yt_brand_impr = data$youtube_brand_impressions,
  yt_performance_spend = data$youtube_performance_spend,
  yt_performance_impr = data$youtube_performance_impressions,
  search_spend = data$search_spend,
  search_clicks = data$search_clicks,
  newsp_spend = data$newspaper_spend,
  newsp_read = data$newspaper_readership,
  tv_spend = data$tv_spend,
  tv_rating = data$tv_gross_rating_points
)

# Split exogenous variables into training and future (if forecasting)
exog_train <- exogenous_vars[1:length(train_ts_data), ]
exog_future <- exogenous_vars[(length(train_ts_data) + 1):(length(train_ts_data) + forecast_horizon), ]

# Fit ARIMAX model
arimax_model <- auto.arima(train_ts_data, xreg = as.matrix(exog_train))

# Forecast with exogenous variables
sales_forecast_exog <- forecast(arimax_model, xreg = as.matrix(exog_future), h = forecast_horizon)

# Plot forecasts
autoplot(sales_forecast_exog, main = "Sales Forecast using ARIMAX")

results_arimax <- data.frame(
  Actual = test_ts_data,
  Predicted = sales_forecast_exog$mean
)

results_arimax$Date <- data$week[126:156]

ggplot(results_arimax, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual Sales")) +
  geom_line(aes(y = Predicted, color = "Predicted Sales"), linetype = "dashed") +
  labs(
    title = "Actual vs. Predicted Sales Over Time",
    x = "Date",
    y = "Sales"
  ) +
  scale_color_manual("", values = c("Actual Sales" = "blue", "Predicted Sales" = "red")) +
  theme_minimal()

summary(arimax_model)

actuals_arimax <- test_ts_data
mae_ax <- mae(actuals_arimax, sales_forecast_exog$mean)
rmse_ax <- rmse(actuals_arimax, sales_forecast_exog$mean)
mape_ax <- mape(actuals_arimax, sales_forecast_exog$mean) * 100  # Convert to percentage

cat("MAE:", mae_ax, "\n")
cat("RMSE:", rmse_ax, "\n")
cat("MAPE:", mape_ax, "%\n")


########################      Multiple Regression       ########################

set.seed(123)
str(data)
training_set <- data[1:125,-c(4,17)]
test_set = data[126:156,-c(4,17)]


regressor = lm(formula = sales ~ .,
               data = training_set)
y_pred = predict(regressor, newdata = test_set)

summary(regressor)

#Plot Predicted vs Actual Sales
results <- data.frame(
  Actual = test_set$sales,
  Predicted = y_pred
)

test_set$week <- as.Date(test_set$week)

results$Date <- test_set$week

ggplot(results, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual Sales")) +
  geom_line(aes(y = Predicted, color = "Predicted Sales"), linetype = "dashed") +
  labs(
    title = "Actual vs. Predicted Sales Over Time",
    x = "Date",
    y = "Sales"
  ) +
  scale_color_manual("", values = c("Actual Sales" = "blue", "Predicted Sales" = "red")) +
  theme_minimal()

actuals_multiple_reg <- test_set$sales
mae_ml <- mae(actuals_multiple_reg, y_pred)
rmse_ml <- rmse(actuals_multiple_reg, y_pred)
mape_ml <- mape(actuals_multiple_reg, y_pred) * 100  # Convert to percentage

cat("MAE:", mae_ml, "\n")
cat("RMSE:", rmse_ml, "\n")
cat("MAPE:", mape_ml, "%\n")


############################      ROBYN LIBRARY     ############################   

########################     Marketing Mix Modeling    #########################

data <- read.csv("Desktop/ROBYN PROJECT/jiali/data_week_final.csv")

set.seed(123)
Sys.setenv(R_FUTURE_FORK_ENABLE = "true") #enable multi core processing, allowing the script to run faster by utilizing multiple CPU cores
create_files <- TRUE

colnames(data)[colnames(data)=='week'] <- 'date'
data$date <- as.Date(data$date, format = '%Y-%m-%d')
data$year <- substr(data$date, 1,4)
data$year <- as.factor(data$year)

data_2 <- data[data$date>='2021-01-01' & data$date<='2022-04-30',]

data("dt_prophet_holidays")

robyn_directory <- '~/Desktop/ROBYN PROJECT'

#Model Specification 4 steps
#1) Input Variables, create InputCollect
InputCollect <- robyn_inputs(
  dt_input = data_2,
  dt_prophet_holidays = dt_prophet_holidays,
  
  #define dependent variable
  dep_var = 'sales',
  dep_var_type = 'revenue',
  date_var = 'date',
  
  #set time variables
  prophet_vars = c('trend','season','holiday'),
  prophet_signs = c('default','default','default'),
  prophet_country = 'US',
  
  #set context variables
  context_vars = c('unemployment','GDP','Income'),
  context_signs = c('default','default','default'),
  factor_vars = c('unemployment','GDP','Income'),
  
  #set media variables
  paid_media_spends = c("facebook_newsfeed_spend", "youtube_brand_spend", "search_spend", "youtube_performance_spend", "newspaper_spend","tv_spend"), 
  paid_media_signs = c('positive','positive','positive','positive','positive','positive'),
  paid_media_vars = c("facebook_newsfeed_impressions", "youtube_brand_impressions", "search_clicks", "youtube_performance_impressions", "newspaper_readership","tv_gross_rating_points"),
  
  cores = NULL,
  
  hyperparameters = NULL,
  
  #rolling window
  window_start = '2021-01-01',
  window_end = '2022-04-30',
  
  adstock = 'weibull_cdf'
)

print(InputCollect)

plot_adstock(plot=TRUE)
plot_saturation(plot=TRUE)

hyper_names(adstock = InputCollect$adstock, all_media = InputCollect$all_media)
hyper_limits()

#Set hyperparamaters
#robyn provides plausible / default boundaries (ranges) for each parameter that the Nevergrad Algorithm will optimize for
hyperparameters <- list(
  facebook_newsfeed_spend_alphas = c(0.5, 3),
  facebook_newsfeed_spend_gammas = c(0.3, 1),
  facebook_newsfeed_spend_shapes= c(0, 2),
  facebook_newsfeed_spend_scales =c(0, 0.1),
  
  newspaper_spend_alphas = c(0.5, 3),
  newspaper_spend_gammas = c(0.3, 1),
  newspaper_spend_shapes= c(0, 2),
  newspaper_spend_scales=c(0, 0.1),
  
  search_spend_alphas = c(0.5, 3),
  search_spend_gammas = c(0.3, 1),
  search_spend_shapes= c(0, 2),
  search_spend_scales=c(0, 0.1),
  
  tv_spend_alphas = c(0.5, 3),
  tv_spend_gammas = c(0.3, 1),
  tv_spend_shapes= c(0, 2),
  tv_spend_scales=c(0, 0.1),
  
  youtube_brand_spend_alphas = c(0.5, 3),
  youtube_brand_spend_gammas = c(0.3, 1),
  youtube_brand_spend_shapes= c(0, 2),
  youtube_brand_spend_scales=c(0, 0.1),
  
  youtube_performance_spend_alphas = c(0.5, 3),
  youtube_performance_spend_gammas = c(0.3, 1),
  youtube_performance_spend_shapes= c(0, 2),
  youtube_performance_spend_scales=c(0, 0.1),
  train_size = c(0.5,0.8)
)

#add hyperparameters into robyn_inputs
InputCollect <- robyn_inputs(InputCollect = InputCollect, hyperparameters = hyperparameters)
print(InputCollect)


##########################     Build initial model      ########################
#Run all trials and iterations. 
#Set output
OutputModels <- robyn_run(
  InputCollect = InputCollect, # feed in all model specification
  cores = NULL, # default to max available
  # add_penalty_factor = FALSE, # Untested feature. Use with caution.
  outputs = FALSE, # outputs = FALSE disables direct model output - robyn_outputs()
  iterations=3000,
  trials=5
)
print(OutputModels)

#export models in files
OutputCollect <- robyn_outputs(
  InputCollect, OutputModels,
  pareto_fronts = "auto",
  # calibration_constraint = 0.1, # range c(0.01, 0.1) & default at 0.1
  csv_out = "pareto", # "pareto" or "all"
  clusters = TRUE, # Set to TRUE to cluster similar models by ROAS. See ?robyn_clusters
  plot_pareto = TRUE, # Set to FALSE to deactivate plotting and saving model one-pagers
  plot_folder = robyn_directory # path for plots export
)

print(OutputCollect)

OutputCollect$allSolutions

select_model <- "4_391_6"

ExportedModelOld <- robyn_write( 
  robyn_directory = robyn_directory, # model object location and name
  select_model = select_model, # selected model ID
  InputCollect = InputCollect, # all model input
  OutputCollect = OutputCollect, # all model output
  export = create_files
)

data_refresh<-data[data$date >= '2021-01-03' & data$date <= '2022-07-01',]

RobynRefresh <- robyn_refresh(
  robyn_directory = robyn_directory,
  dt_input = data_refresh,
  dt_holidays = dt_prophet_holidays,
  refresh_steps = 9,
  refresh_mode= 'auto',
  refresh_iters = 1000, # 1k is an estimation
  refresh_trials = 3,
  clusters=TRUE
)

print(ExportedModelOld)

myOnePager <- robyn_onepagers(InputCollect, OutputCollect, select_model, export = create_files)

#BUDGET ALLOCATION BASED ON THE SELECTED MODEL
AllocatorCollect1 <- robyn_allocator(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect,
  select_model = select_model,
  scenario = "max_historical_response",  #what's the most I can get in terms of revenue by changing the allocation of budgets
  channel_constr_low = c(0.5, 0.8, 0.5,0.8, 0.8, 0.8), 
  channel_constr_up = c(1.1, 1.5, 1.1, 1.4, 1.6, 1.7),
  export = create_files
)

print(AllocatorCollect1$total_budget)

# Predict with 6590449 Budget Allocation
AllocatorCollect2 <- robyn_allocator(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect,
  select_model = select_model,
  scenario = "max_response",
  channel_constr_low = c(0.5, 0.8, 0.5,0.8, 0.8, 0.8),
  channel_constr_up = c(1.1, 1.5, 1.1, 1.4, 1.6, 1.7),
  expected_spend = 6590449, # Total spend to be simulated
  expected_spend_days = 120, # Duration of expected_spend in days
  export = create_files
)
print(AllocatorCollect2)

# Scenario "target_efficiency": "How much to spend to hit ROAS or CPA of x?"
AllocatorCollect3 <- robyn_allocator(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect,
  select_model = select_model,
  date_range = NULL, # Default last month as initial period
  scenario = "target_efficiency",
  target_value = 1, # Customize target ROAS or CPA value
  export = create_files
)
print(AllocatorCollect3)

InputCollect$all_media
select_media <- list("tv_spend", "youtube_brand_spend", "newspaper_spend")
M <- length(select_media)
metric_value <- numeric()

#For paid_media_spends set metric_value as your optimal spend
for(i in 1:M){
  metric_value[i] <- AllocatorCollect1$dt_optimOut$optmSpendUnit[
    AllocatorCollect1$dt_optimOut$channels == select_media[i]]
  optimal_spend <- data.frame(select_media[i], metric_value[i])
  print(optimal_spend)
}







