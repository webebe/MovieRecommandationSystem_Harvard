################################
# Create edx set, validation set
################################

library(tidyverse)
library(dslabs)
library(HistData)
library("ggpubr")
library(broom)
library(caret)
library(lubridate)
library(purrr)
library(pdftools)
library(matrixStats)
library(dplyr)
library(randomForest)
library(Rborist)
library(rpart)
library(rpart.plot)
library(gam)
library(ggplot2)
library(lubridate)
library(RColorBrewer)
library(gam)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#define RMSE function to validate
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
#define gloabal mean rating
mu <- mean(edx$rating)
mu
#Set baseline model for comparison
naive_rmse <- RMSE(validation$rating, mu)
rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)

#Visualize movie effect
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")
#visualize User effect
edx %>% dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")
#visualize genre effect
edx %>% dplyr::count(genres) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("genres")

#Movie effect model
mu <- mean(edx$rating) 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 30, data = ., color = I("black"))
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i
model_movie_rmse <- RMSE(validation$rating,predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_movie_rmse ))
rmse_results %>% knitr::kable()

#User effect model
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i))
user_avgs %>% qplot(b_u, geom="histogram", bins = 30, color = "black", data= .)

#Movie and User model
predicted_ratings_user_movie <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs,  by='userId') %>%
  mutate(pred = mu + b_i + b_u) 
# test and save rmse results 
model_user_movie_rmse <- RMSE(validation$rating,predicted_ratings_user_movie$pred)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie and User Effect Model",  
                                     RMSE = model_user_movie_rmse ))

#regularization
#Create an additional partition of training and test sets from the provided edx dataset
#Cross validation must not be performed on the validation dataset!
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# Choose lambda by cross validation.
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(lambda){
   b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(test_set$rating,predicted_ratings))
})
# Plot rmses vs lambdas to see which lambda minimizes rmse
qplot(lambdas, rmses)  
lambda<-lambdas[which.min(rmses)]

# Movie effect regularized b_i using lambda
movie_avgs_reg <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())
# User effect regularized b_u using lambda
user_avgs_reg <- edx %>% 
  left_join(movie_avgs_reg, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda), n_u = n())
# Predict ratings
predicted_ratings_reg <- validation %>% 
  left_join(movie_avgs_reg, by='movieId') %>%
  left_join(user_avgs_reg, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>% 
  .$pred

model_reg_rmse <- RMSE(validation$rating,predicted_ratings_reg)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie and User Model",  
                                     RMSE = model_reg_rmse ))
rmse_results %>% knitr::kable()

#genre model
# b_y and b_g represent the year & genre effects, respectively
lambdas2 <- seq(0, 20, 1)
rmses <- sapply(lambdas2, function(lambda2){
 
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda2))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda2))
  
 b_g <- edx %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+lambda2), n_g = n())
  
  predicted_ratings <- validation %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_g, by = 'genres') %>%
    mutate(pred = mu + b_i + b_u + b_g) %>% 
    .$pred
  
  return(RMSE(validation$rating,predicted_ratings))
})
# Compute new predictions using the optimal lambda
# Test and save results 
qplot(lambdas2, rmses) 
lambda2<-lambdas2[which.min(rmses)]

#build model
movie_reg_avgs_2 <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda2), n_i = n())
user_reg_avgs_2 <- edx %>% 
  left_join(movie_reg_avgs_2, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda2), n_u = n())
genre_reg_avgs <- edx %>%
  left_join(movie_reg_avgs_2, by='movieId') %>%
  left_join(user_reg_avgs_2, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+lambda2), n_g = n())

predicted_ratings <- validation %>% 
  left_join(movie_reg_avgs_2, by='movieId') %>%
  left_join(user_reg_avgs_2, by='userId') %>%
  left_join(genre_reg_avgs, by = 'genres') %>%
  mutate(pred = mu + b_i + b_u + b_g) %>% 
  .$pred
model_UserMovieGenreReg_rmse <- RMSE(validation$rating,predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Reg Movie, User and Genre Effect Model",  
                                     RMSE = model_UserMovieGenreReg_rmse ))
rmse_results %>% knitr::kable()



mean(edx$rating)
mean(validation$rating)
