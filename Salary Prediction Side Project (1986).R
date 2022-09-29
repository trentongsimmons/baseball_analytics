# Read in the data
library(tidyverse)
library(car)
hitters <- read.csv("https://raw.githubusercontent.com/trentongsimmons/baseball_analytics/main/Hitters.csv")

# Set the appropriate variables to factors
cols <- c(14, 15, 20)
hitters[cols] <- lapply(hitters[cols], factor)

# Flag columns with missing values on the response variable
hitters <- hitters %>%
  mutate(missing = ifelse(is.na(Salary), 1, 0))

# Impute missing Salary data with median
hitters$Salary <- replace(hitters$Salary, is.na(hitters$Salary), median(hitters$Salary, na.rm = TRUE))

# View the distribution of variables individually
par(mfrow = c(1,2))
for (i in 1:20) {
  if (class(hitters[, i]) == "numeric" | class(hitters[, i]) == "integer") {
    qqnorm(hitters[, i], main = paste("QQ Plot of", colnames(hitters[i])))
    hist(hitters[, i], main = paste("Histogram of", colnames(hitters[i])),
                       xlab = colnames(hitters[i]))
  }
}
par(mfrow = c(1,1))

# View correlation matrix
cor_matrix <- cor(hitters[, -c(14, 15, 20)])

# Separate models into training and test datasets
set.seed(0885)
hitters$id <- 1:nrow(hitters)
hitters_train <- hitters %>% dplyr::sample_frac(0.7)
hitters_test <- dplyr::anti_join(hitters, hitters_train, by = 'id')

# Begin with a preliminary model with only those variables with a weak correlation
# to avoid collinearity
lm_prelim <- lm(Salary ~ RBI + CHmRun + CRBI + League + Division + 
                PutOuts + Assists + Errors + NewLeague + missing, data = hitters_train)
summary(lm_prelim)
vif(lm_prelim)
# Little evidence of collinearity

# Explore transformations and take appropriate action
boxCox(lm_prelim)
hitters_train$trans_Salary <- hitters_train$Salary^(1/5)

# Model with transformed response variable
lm_trans <- lm(trans_Salary ~ RBI + CHmRun + CRBI + League + Division + 
               PutOuts + Assists + Errors + NewLeague + missing, data = hitters_train)
summary(lm_trans)

# Explore all potential two-way interactions
lm_int <- lm(trans_Salary ~ RBI + CHmRun + CRBI + League + Division + 
               PutOuts + Assists + Errors + NewLeague + missing +
               RBI*CHmRun + RBI*CRBI + RBI*League +
               RBI*Division + RBI*PutOuts + RBI*Assists + RBI*Errors + RBI*NewLeague +
               RBI*missing + CHmRun*CRBI + CHmRun*League + CHmRun*Division + CHmRun*PutOuts +
               CHmRun*Assists + CHmRun*Errors + CHmRun*NewLeague + CHmRun*missing + CRBI*League +
               CRBI*Division + CRBI*PutOuts + CRBI*Assists + CRBI*Errors + CRBI*NewLeague +
               CRBI*missing + League*Division + League*PutOuts + League*Assists + League*Errors +
               League*NewLeague + League*missing + Division*PutOuts + Division*Assists + 
               Division*Errors + Division*NewLeague + Division*missing + PutOuts*Assists + 
               PutOuts*Errors + PutOuts*NewLeague + PutOuts*missing + Assists*Errors + 
               Assists*NewLeague + Assists*missing + Errors*NewLeague + Errors*missing + 
               NewLeague*missing + I(HmRun^2) + I(RBI^2) + I(CHmRun^2) + I(CRBI^2) +
               I(PutOuts^2) + I(Assists^2) + I(Errors^2), data = hitters_train)

# Complete forward selection technique with AIC
lm_empty <- lm(trans_Salary ~ 1, data = hitters_train)
lm_forward <- step(lm_empty, scope = list(lower = lm_empty, upper = lm_int),
                   direction = "forward", k = 2)
summary(lm_forward)

# Complete backward selection technique with AIC
lm_backward <- step(lm_int, scope = list(lower = lm_empty, upper = lm_int),
                    direction = "backward", k = 2)
summary(lm_backward)

# Complete stepwise selection technique with AIC
lm_stepwise <- step(lm_empty, scope = list(lower = lm_empty, upper = lm_int),
                    direction = "both", k = 2)
summary(lm_stepwise)

# Check AIC
AIC(lm_forward)
AIC(lm_backward)
AIC(lm_stepwise)

# Backward model performs the best, append predictions
hitters_train <- cbind(hitters_train, pred = lm_backward$fitted.values)

# Check modeling assumptions
# Constant variance
ggplot(lm_backward, aes(x = trans_Salary, y = lm_backward$residuals)) +
                          geom_point(color = "blue", size = 3) +
                          labs(title = "Residual Plot", x = "Transformed Salary",
                               y = "Residuals")
cor.test(abs(lm_backward$residuals), lm_backward$fitted.values, method = "spearman", exact = TRUE)
# Appears to be increasing linearly, but numeric test suggests insufficient evidence of constant variance

# Normality
hist(lm_backward$residuals, main = "Histogram of Model Residuals", xlab = "Model Residuals")
# One outlier skewing the distribution, all else looks normally distributed

# Studentized Residuals
n.index = seq(1, nrow(hitters_train))
ggplot(lm_backward, aes(x = n.index, y = rstudent(lm_backward))) +
  geom_point(color = "orange") + geom_line(y = -3) + geom_line(y = 3) +
  labs(title = "External Studentized Residuals", x = "Observations", y = "Residuals")

# Cook's D
d.cut <- 4/(nrow(hitters_train) - length(lm_backward$coefficients) - 1)
ggplot(lm_backward, aes(x = n.index, y = cooks.distance(lm_backward))) +
  geom_point(color = "orange") + geom_line(y = d.cut) + labs(title = "Cook's D",
                                                             x = "Observation",
                                                             y = "Cook's Distance")
hitters_train[which.max(cooks.distance(lm_backward)),]
# Upon closer investigation, this observation is not a misrecording because
# these numbers belong to Tony Perez. As a result, we leave it in the model
# because it still provides valuable information.

# Predict on test dataset and report MAE
hitters_test$trans_Salary <- hitters_test$Salary^(1/5)
test_fit <- predict(lm_backward, newdata = hitters_test)
test_fit_adj <- test_fit^(5)
MAE <- mean(abs(hitters_test$Salary - test_fit_adj))
sd(hitters_train$Salary)
# MAE returns a value of 221.331, indicating a good model because the standard deviation
# of the training dataset is 434.26. Salary is reported in thousands of dollars, meaning
# that model predictions were off on average by $221,331. Although this data is from 1986,
# the same variables are likely to be important in modeling salary today, meaning that career 
# home runs and career/yearly RBI's are important offensive metrics in modeling salary, as well as the 
# defensive metrics of assists, put outs, and errors. Division location was also important, and many
# of the variables included also have significant interactions with one another.
