#----------------------------------------------------------------------------------
# Dataset preparation
#----------------------------------------------------------------------------------

# Required packages

if(!require(tidyverse)) 
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) 
  install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) 
  install.packages("data.table", repos = "http://cran.us.r-project.org")

# File download

if (!file.exists('sdd.csv')) {
  download.file(
    'http://www.stat.columbia.edu/~gelman/arm/examples/speed.dating/Speed%20Dating%20Data.csv', 
    'sdd.csv'
  )
}

# Read data
rawdata <- fread('sdd.csv')
dim(rawdata)

# We create a data frame with the relevant data for each participant
# as described in the PDF of the project

participants <- rawdata %>% distinct(
  iid, gender, age, race, imprace, imprelig, career_c, 
  sports, tvsports, exercise, dining, museums, art, hiking, gaming, clubbing, 
  reading, tv, theater, movies, concerts, music, shopping, yoga,   
  attr1_1, sinc1_1, intel1_1, fun1_1, amb1_1, shar1_1,
  attr3_1, sinc3_1, intel3_1, fun3_1, amb3_1
)

# Remove participants with missing values
participants <- participants[rowSums(is.na(participants)) == 0,]

# Create a data frame with the outcome (match),
# the data gathered for the first participant,
# and the data gathered for the second participant.

dates <- rawdata %>% 
  filter(gender == 0) %>%
  select(iid, pid, match) %>% 
  inner_join(participants, by='iid') %>% 
  inner_join(participants, by=c('pid' = 'iid')) %>%
  select(-c(iid, pid))

rm(rawdata, participants)


#### THE CODE WAS TESTED USING R VERSION 3.2.3, PAY ATTENTION TO THE SEED ####
set.seed(1) # if using R higher than 3.5 , use `set.seed(1, sample.kind="Rounding")`

# Our dataset has a high prevalence of non-match dates. 
# We are going to discard random non-matches to equalize the ratio:

# Separate matches and non-marches
matches = dates %>% filter(match == 1)
nonmatches = dates %>% filter(match == 0)

# and join all the matches and an equal number of (random) non-matches
dates <- rbind(matches, nonmatches[sample(nrow(nonmatches), nrow(matches)), ])

rm(matches, nonmatches)



# Now we are going to split the data into a training set and a validation set.

test_index <- createDataPartition(y = dates$match, times = 1, p = 0.1, list = FALSE)
training <- dates[-test_index,]
validation <- dates[test_index,]
rm(dates, test_index)

#---------------------------------------------------------------------------------
# Analysis
#---------------------------------------------------------------------------------

# Function that computes the features describen in the PDF of the project
extract_features <- function(df) {
  
  # lets compute the shared interests as a separate data frame to use it later
  shared_interests <- sqrt(
    (df$sports.x    - df$sports.y)^2 +
    (df$tvsports.x  - df$tvsports.y)^2 +
    (df$exercise.x  - df$exercise.y)^2 +
    (df$dining.x    - df$dining.y)^2 +
    (df$museums.x   - df$museums.y)^2 +
    (df$art.x       - df$art.y)^2 +
    (df$hiking.x    - df$hiking.y)^2 +
    (df$gaming.x    - df$gaming.y)^2 +
    (df$clubbing.x  - df$clubbing.y)^2 +
    (df$reading.x   - df$reading.y)^2 +
    (df$tv.x        - df$tv.y)^2 +
    (df$theater.x   - df$theater.y)^2 +
    (df$movies.x    - df$movies.y)^2 +
    (df$concerts.x  - df$concerts.y)^2 +
    (df$music.x     - df$music.y)^2 +
    (df$shopping.x  - df$shopping.y)^2 +
    (df$yoga.x      - df$yoga.y)^2
  )

  return (
    # compute the features
    df %>% mutate(
      agediff = age.x - age.y,
      srace.x = ifelse(race.x == race.y, 1, -1) * imprace.x,
      srace.y = ifelse(race.x == race.y, 1, -1) * imprace.y,
      attr.x = attr3_1.x * attr1_1.y,
      attr.y = attr3_1.y * attr1_1.x,
      sinc.x = sinc3_1.x * sinc1_1.y,
      sinc.y = sinc3_1.y * sinc1_1.x,
      fun.x = fun3_1.x * fun1_1.y,
      fun.y = fun3_1.y * fun1_1.x,
      intel.x = intel3_1.x * intel1_1.y,
      intel.y = intel3_1.y * intel1_1.x,
      amb.x = amb3_1.x * amb1_1.y,
      amb.y = amb3_1.y * amb1_1.x,
      shared.x = shared_interests * shar1_1.y,
      shared.y = shared_interests * shar1_1.x
    ) %>% select(
      # and return only the result and the features
      match, agediff, srace.x, srace.y, imprelig.x, imprelig.y, 
      career_c.x, career_c.y, attr.x, attr.y, sinc.x, sinc.y, 
      fun.x, fun.y, intel.x, intel.y, amb.x, amb.y, shared.x, shared.y
    )
  )
  
}

# extract the features from training and validation set
feat_training <- extract_features(training)
feat_validation <- extract_features(validation)

# Fit a linear model using the training features and outcomes
model1 <- lm(match ~ agediff + srace.x + srace.y + imprelig.x + imprelig.y +
            career_c.x + career_c.y + attr.x + attr.y + sinc.x + sinc.y +
             fun.x + fun.y + intel.x + intel.y + amb.x + amb.y + shared.x + shared.y, 
             data = feat_training)

# Predict the outcome using only the training features
match_hat <- feat_training %>% 
  select(-match) %>% 
  mutate(match_hat = predict(model1, newdata = .)) %>% 
  pull(match_hat)
match_hat <- ifelse(match_hat >= 0.5, 1, 0)

# Compute the in-sample accuracy
mean(match_hat == feat_training$match)

# Predict the outcome using only the validation features
match_hat <- feat_validation %>% 
  select(-match) %>% 
  mutate(match_hat = predict(model1, newdata = .)) %>% 
  pull(match_hat)
match_hat <- ifelse(match_hat >= 0.5, 1, 0)

# Compute the out-sample accuracy
mean(match_hat == feat_validation$match)

# We are going to try a more complex model: the k-nearest neighbour algorithm:

# We need to convert the outcomes to factors
feat_validation$match <- factor(feat_validation$match)
feat_training$match <- factor(feat_training$match, levels = levels(feat_validation$match))

# Let's train the model on the training set
train_knn <- train(
  x = feat_training %>% select(-match), 
  feat_training$match, 
  method='knn', 
  tuneGrid = data.frame(k = seq(5, 21, 2))
)

# Then compute the output for the training set
match_hat_knn <- predict(train_knn, feat_training %>% select(-match), type = "raw")

# And display the in-sample accuracy
mean(match_hat_knn == feat_training$match)

# Then compute the output for the validation set
match_hat_knn <- predict(train_knn, feat_validation %>% select(-match), type = "raw")

# And display the out-sample accuracy
mean(match_hat_knn == feat_validation$match)

# We suspect that this model is overfitting.
# Let's try another algorithm: random forests.

# Let's train the model on the training set
train_rf  <- train(
  x = feat_training %>% select(-match), 
  feat_training$match, 
  method='rf',
  trControl = trainControl(number = 10)
)

# Then compute the output for the training set
match_hat_rf  <- predict(train_rf,  select(feat_training, -match), type = "raw")

# And display the in-sample accuracy
mean(match_hat_rf  == feat_training$match)

# The accuracy in the training set is 1. 
# The random forest can make enouh partitions to fit all the data.
# It is not generalizing so we expect a big drop in out-sample performance:

# Compute the output for the validation set
match_hat_rf  <- predict(train_rf,  feat_validation %>% select(-match), type = "raw")

# And display the out-sample accuracy
mean(match_hat_rf  == feat_validation$match)

# Let's see if the random forest is able to generalize when fed with the inicial training set before feature extraction.

# We need to convert the outcomes to factors
validation$match <- factor(validation$match)
training$match <- factor(training$match, levels = levels(validation$match))

# Let's train the model on the training set
train_rf_raw  <- train(
  x = training %>% select(-match), 
  training$match, 
  method='rf',
  trControl = trainControl(number = 10)
)

# Then compute the output for the validation set
match_hat_rf_raw  <- predict(train_rf_raw,  training %>% select(-match), type = "raw")

# And display the in-sample accuracy
mean(match_hat_rf_raw  == training$match)

# The random forest still can learn the entire training dataset.

# Let's compute the output for the validation set
match_hat_rf_raw  <- predict(train_rf_raw,  validation %>% select(-match), type = "raw")

# And display the out-sample accuracy
mean(match_hat_rf_raw  == validation$match)

# It seems that kind of models are not suitable for aur data.
# They tend to overfit, the performance is very similar
# and the linear model has an added value of interpretability

model1

# We can find insights analyzing the cofficients of each feature
# and continue to improve the model.