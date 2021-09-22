################################################################################
###########                                                            #########
#                                  WARNING                                     #
###########                                                            #########  
################################################################################
# Do not simply source this project without reading!                           #
# I used parallel cloud computing for several of my models which could crash R #
# if you run them locally on a system that doesn't meet requirements for CPU   #
# threads and ram and gpu. (I used AWS - CPU 32 threads, 64gb Ram, 8gb GPU) I  #
# I have provided detailed notes about the run time and system requirements    #
# for each of those models as we approach them.                                #
################################################################################

# SHHHHH BE QUIET
options(warning=FALSE, message=FALSE)

#LIBRARIES
if(!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret))
  install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(readxl))
  install.packages("readxl", repos = "http://cran.us.r-project.org")
if(!require(randomForest))
  install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(matrixStats))
  install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(GGally))
  install.packages("GGally", repos = "http://cran.us.r-project.org")
if(!require(doParallel))
  install.packages("doParallel", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(readxl)
library(randomForest)
library(matrixStats)
library(GGally)
library(doParallel)


# PULL THE DATA
tmp <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip", tmp)

dat <- read_excel(unzip(tmp, "DryBeanDataset/Dry_Bean_Dataset.xlsx"))

glimpse(dat)

anyNA(dat)

# Transform the data object to a matrix of x features and y factors
db <- with(dat, list(x =as.matrix(dat[-17]), y = as.factor(dat$Class)))

class(db$x)

class(db$y)

# double checking that the observations were maintained
dim(db$x)[1]
dim(db$x)[2]

# Check how the beans are represented on average in the dataset
mean(db$y == "BOMBAY")
mean(db$y == "SEKER")
mean(db$y == "CALI")
mean(db$y == "DERMASON")
mean(db$y == "HOROZ")
mean(db$y == "SIRA")
mean(db$y == "BARBUNYA")


#Pairwise Correlation matrix using Pearson method, default method is Pearson
db$x %>% ggcorr(high = "orange",
                low = "green",
                label = TRUE, 
                hjust = .9, 
                size = 2, 
                label_size = 2,
                nbreaks = 5) +
  labs(title = "Correlation Matrix", 
       subtitle = "Pearson Method Using Pairwise Obervations")

# As we would expect, we can see there are some very high correlations in 
# features, particularly among the geometric measurement features like area and 
# perimeter. 


# Which feature has the highest mean
which.max(colMeans(db$x))

# Which feature has the lowest standard deviation
which.min(colSds(db$x))

# Center and scale the data
x_centered <- sweep(db$x, 2, colMeans(db$x))
x_scaled <- sweep(x_centered, 2, colSds(db$x), FUN = "/")

# checking sd and median
sd(x_scaled[,1])
median(x_scaled[,1])

# calculates the distance between all samples using the scaled matrix. 
d_samples <- dist(x_scaled)

# calculates the distance between the first sample which is classified as Seker 
# and other Seker samples.
dist_SEtoSE <- as.matrix(d_samples)[1, db$y == "SEKER"]
mean(dist_SEtoSE[2:length(dist_SEtoSE)])

# Calculates distance from the first Seker sample to Bombay samples. 
dist_SEtoBO <- as.matrix(d_samples)[1, db$y == "BOMBAY"]
mean(dist_SEtoBO)

# makes a heatmap of the relationship between features using the scaled 
# matrix.
d_features <- dist(t(x_scaled))
heatmap(as.matrix(d_features))

# performs hierarchichal clustering of the 16 features 
h <- hclust(d_features)

# cuts the tree to 5 groups
groups <- cutree(h, k = 5)
split(names(groups), groups)

# Principal Component Analysis
pca <- prcomp(x_scaled)
pca_sum <- summary(pca)
pca_sum$importance

# Compute Variance explained and proportion of PC's
var_explained <- cumsum(pca$sdev^2/sum(pca$sdev^2))

plot(var_explained, xlab = "Principal Components",
     ylab = "Cumulative Proportion of Variance Explained", type = "b")

# Visualize the dilineation of classes from the 1st two PC's
data.frame(pca$x[,1:2], type = db$y) %>%
  ggplot(aes(PC1, PC2, value, color = type)) +
  geom_point(alpha = .5)

# That plot tells us that Bombay beans tend to have very high values of PC1
# and PC2. It also tells us, among other things, that Sira beans have a similar 
# spread of values for PC1 and PC2.

# Makes a boxplot of the first 5 PCs grouped by bean type
data.frame(type = db$y, pca$x[,1:5]) %>%
  gather(key = "PC", value = "value", -type) %>%
  ggplot(aes(PC, value, fill = type)) +
  geom_boxplot()


# new db with feature reduction from pca
db <- with(db, list(x =as.matrix(pca$x[,1:7]), y = as.factor(db$y)))

# Split the data 80% train 20% test
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(db$y, times = 1, p = 0.2, list = FALSE)
test_x <- db$x[test_index,]
test_y <- db$y[test_index]
train_x <- db$x[-test_index,]
train_y <- db$y[-test_index]

## LDA model
set.seed(12, sample.kind = "Rounding")
train_lda <- train(train_x, train_y, method = "lda", 
                   trControl = trainControl(method = "cv", number = 5 ))

# reported accuracy from the cross validation in the train set only
lda_acc <- train_lda$results[,2]

Accuracy_Results <- tibble(Method = "LDA", Accuracy = lda_acc)
Accuracy_Results %>% knitr::kable()

# QDA Model
set.seed(312, sample.kind = "Rounding")
train_qda <- train(train_x, train_y, method = "qda", 
                   trControl = trainControl(method = "cv", number = 5 ))

# reported Accuracy from the cross validation in the train set only
qda_acc <- train_qda$results[,2]

Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "QDA", Accuracy = qda_acc))
Accuracy_Results %>% knitr::kable()

# Treebag bootstrap Aggregating Model
set.seed(9874, sample.kind="Rounding")
trCtrl <- trainControl(method = "cv", number = 5)
cr.fit <- train(train_x, train_y,
                method  = "treebag",
                trControl = trCtrl,
                metric = "Accuracy")

# reported Accuracy from the cross validation in the train set only
crtb_acc <- cr.fit$results[,2]

Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "TREEBAG", Accuracy = crtb_acc))
Accuracy_Results %>% knitr::kable()


#SVM Linear Grid Model
################################################################################
# IMPORTANT NOTE - I used cloud computing on AWS                               #
# this TAKES 3 minutes for my setup which had a 32 thread CPU, 32gb ram and a  #
# gpu with 8gb of dedicated memory. It didn't come close to taxing anything.   #
# Therefore this should be safe to run locally.                                #
#                                                                              #
# It will take you a longer run time but it wont crash.  This is because the   #
# cross validation is run in parallel. There is not a massive ram requirement. #    
################################################################################

set.seed(56543, sample.kind="Rounding")
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3) 
grid <- expand.grid(C = seq(0.85, .95, 0.01))
svm_Linear_Grid <- train(train_x, train_y, 
                         method = "svmLinear",
                         trControl=trctrl,
                         tuneGrid = grid,
                         tuneLength = 10)

# Print the best tuning parameter C that
# maximizes model accuracy

svm_Linear_Grid$bestTune

plot(svm_Linear_Grid)

# reported Accuracy from the cross validation in the train set only
svm_LG_Acc <- svm_Linear_Grid$results[3,2]

Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "LinearSVM", 
                                     Accuracy = svm_LG_Acc))
Accuracy_Results %>% knitr::kable()

# SVM RADIAL
################################################################################
# IMPORTANT NOTE - I used cloud computing on AWS                               #
# this TAKES 8 minutes for my setup which had a 32 thread CPU, 32gb ram and a  #
# GPU with 8gb of dedicated memory. This really only taxed the GPU - it used   #
# 1gb of dedicated GPU Memory. Therefore this should be safe to run locally if #
# you have a modern machine with a dedicated graphics card.                    #
#                                                                              #    
# The time is because it uses repeated cross validation to choose the sigma    # 
# and Cost parameters that maximize the model accuracy.                        # 
################################################################################

set.seed(1985, sample.kind = "Rounding")
radial_svm <- train(train_x, train_y, 
                    method = "svmRadial", 
                    trControl = trainControl("repeatedcv", 
                                             number = 10, repeats = 3),
                    tuneLength = 10)
# Print the best values for sigma and cost and plot the model
# accuracy according to the Cost parameter.

radial_svm$bestTune
plot(radial_svm)

# reported Accuracy from the cross validation in the train set only
radial_svm_acc <- radial_svm$results[5,3]

Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "RadialSVM", 
                                     Accuracy = radial_svm_acc))
Accuracy_Results %>% knitr::kable()



# RANDOM FOREST WITH PARALLEL Cloud COMPUTING TO SAVE ON TIME

####################### DO NOT RUN #############################################
#    unless you are certain your machine is capable          ###################
#   With 32 threaded CPU it used 34 gb of Ram                       ############
################################################################################
# The Parallel Computing is why it is demanding in RAM                         #
# But it takes too long to run without it.                                     #
################################################################################

numbCores <- detectCores()-4 # Protects against a crash & makes my core count 28
cl <- makeCluster(numbCores)
registerDoParallel(cl)

set.seed(9, sample.kind="Rounding")
tuning <- data.frame(mtry = c(1,2,3,4,5))
train_rf <- train(train_x, train_y,
                  method = "rf",
                  tuneGrid = tuning,
                  importance = TRUE)

# Show the best tune.
train_rf$bestTune


# Next, we take a look at the variable importance.
plot(varImp(train_rf))

# Report the accuracy of the best tune from cross validation in the train set
rf_acc <- train_rf$results[2,2]
Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "RandomForest", 
                                     Accuracy = rf_acc))
Accuracy_Results %>% knitr::kable()

# KNN MODEL
set.seed(234, sample.kind="Rounding")
tuning <- data.frame(k = seq(20, 30, 1))
train_knn <- train(train_x, train_y,
                   method = "knn", 
                   tuneGrid = tuning)

# Pull the best tune.
train_knn$bestTune


# Plot the model accuracy of each k neighbors.
plot(train_knn)

# report the accuracy from the cross validation in the train set
knn_acc <- train_knn$results[4,2]


Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "knn", Accuracy = knn_acc))
Accuracy_Results %>% knitr::kable()

# NN Model
#####################  Warning  ################################################
# The parallel cluster is still running for this model
# It is not a ram heavy model and can be run with a regular modern cpu
# Just be prepared to wait if you are not able implement the 
# doParallel library as shown above 

# One more reminder - I used AWS with a 32 thread CPU and the parallel 
# computations still take a few minutes - I could have listed all the algorithms
# with the hyper-parameters from the best tune... but then you wouldn't know 
# what method I used to find them
################################################################################

set.seed(2976, sample.kind = "Rounding")
# I will use cross validation in the train set in order to find the optimal 
# hidden layers and decay.
tc <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
cv_nn <- train(train_x, train_y, method='nnet', linout=TRUE, trace = FALSE, 
               trControl = tc,
               
               #Grid of tuning parameters to try:
               tuneGrid=expand.grid(.size= seq(15,25,1),.decay=seq(0.15,0.25,0.01))) 

# Examine the results with a plot
cv_nn$bestTune
plot(cv_nn)

# report the accuracy from the cross validation and append it to the running
# table
nn_acc <- cv_nn$results[52,3]

# Appends results to the table
Accuracy_Results <- bind_rows(Accuracy_Results,
                              tibble(Method = "nnet", Accuracy = nn_acc))
Accuracy_Results %>% knitr::kable()

## FINAL VALIDATION NN MODEL
# use the cross validated nnet model to run the predictions on the test set

preds_nn <- predict(cv_nn, test_x)

# Calculates Accuracy

final_Val <- tibble(Method = "Final Val - NNet", Accuracy = mean(preds_nn==test_y))
final_Val

# SUMMARY
confusionMatrix(preds_nn,test_y)

# stop cluster
stopCluster(cl)
