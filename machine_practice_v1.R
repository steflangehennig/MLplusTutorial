######################################################################
# tutorial
# https://www.machinelearningplus.com/machine-learning/caret-package/
######################################################################

install.packages(c('caret', 'skimr', 'RANN', 'randomForest', 'fastAdaboost', 'gbm', 'xgboost', 'caretEnsemble', 'C50', 'earth'))

library(caret)

orange <- read.csv('https://raw.githubusercontent.com/selva86/datasets/master/orange_juice_withmissing.csv')

str(orange)

head(orange[, 1:10])

# Create the training and test datasets
set.seed(100)

# Step 1: Get row numbers for the training data
trainRowNumbers <- createDataPartition(orange$Purchase, p=0.8, list=FALSE)

# Step 2: Create the training  dataset
trainData <- orange[trainRowNumbers,]

# Step 3: Create the test dataset
testData <- orange[-trainRowNumbers,]

# Store X and Y for later use.
x = trainData[, 2:18]
y = trainData$Purchase

# descriptive stats
library(skimr)
skimmed <- skim_to_wide(trainData)
skimmed[, c(1:5, 9:11, 13, 15:16)]


# imputing missing data (pre-processing)
# Create the knn imputation model on the training data
preProcess_missingdata_model <- preProcess(trainData, method='knnImpute')
preProcess_missingdata_model

# use model to predict values of missing data in training data
library(RANN)  # required for knnInpute
trainData <- predict(preProcess_missingdata_model, newdata = trainData)
anyNA(trainData)

# one hot encoding for dummy vars (for non-numerical)
# Creating dummy variables is converting a categorical variable to as many binary variables as here are categories.
dummies_model <- dummyVars(Purchase ~ ., data=trainData)

# Create the dummy variables using predict. The Y variable (Purchase) will not be present in trainData_mat.
trainData_mat <- predict(dummies_model, newdata = trainData)

# # Convert to dataframe
trainData <- data.frame(trainData_mat)

# See the structure of the new dataset
  # store7 was encoded so now there's a "store7.no" and "store7.yes"
str(trainData)

# ranging (normalizing?) the data
preProcess_range_model <- preProcess(trainData, method='range')
trainData <- predict(preProcess_range_model, newdata = trainData)

# Append the Y variable
trainData$Purchase <- y

apply(trainData[, 1:10], 2, FUN=function(x){c('min'=min(x), 'max'=max(x))})

# plotting to see what vars may be important to predicting the outcome
  # box
featurePlot(x = trainData[, 1:18], 
            y = trainData$Purchase, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))
  # density
featurePlot(x = trainData[, 1:18], 
            y = trainData$Purchase, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

# selecting important features to determine "important" vars is the recursive feature elimination (RFE) 
# use most important vars to model on subsets, rank them, iterate through again
### THIS MAY NOT WORK ###
  # make sure 'CVST' and 'recipes' packages are downloaded. 
install.packages("randomForest")
library(randomForest)
set.seed(100)
options(warn=-1)

subsets <- c(1:5, 10, 15, 18)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x=trainData[, 1:18], y=trainData$Purchase,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile

# See available algorithms in caret
modelnames <- paste(names(getModelInfo()), collapse=',  ')
modelnames

modelLookup('earth')

# Set the seed for reproducibility
set.seed(100)

# Train the model using randomForest and predict on the training data itself.
model_mars = train(Purchase ~ ., data=trainData, method='earth')
fitted <- predict(model_mars)

# basic CV and hyper parameter tuning already done...
# The chosen model and its parameters is reported in the last 2 lines of the output (nprune=11, degree=1 here)
model_mars

# plot model to show various iterations of hyperparameter search performed (again, 11 looks highest)
plot(model_mars, main="Model Accuracies with MARS")

# computing importance of variables (variable importance)-which vars are useful?
varimp_mars <- varImp(model_mars)
plot(varimp_mars, main="Variable Importance with MARS")

# preparing test dataset and prediction
# Missing Value imputation -> One-Hot Encoding -> Range Normalization
  # You need to pass the testData through these models in the same sequence

# Step 1: Impute missing values 
testData2 <- predict(preProcess_missingdata_model, testData)  

# Step 2: Create one-hot encodings (dummy variables)
testData3 <- predict(dummies_model, testData2)

# Step 3: Transform the features to range between 0 and 1
testData4 <- predict(preProcess_range_model, testData3)

# View
head(testData4[, 1:10])

# predict on test data
predicted <- predict(model_mars, testData4)
head(predicted)

# computing confusion matrix to compare predictions (data) to the actuals (reference)
  # "everything" provides most classification eval metrics
confusionMatrix(reference = testData$Purchase, data = predicted, mode='everything', positive='MM')


# hyperparameter tuning to optimize model for better performance

  # setting up trainControl
fitControl <- trainControl(
  method = 'cv',                   # k-fold cross validation
  number = 5,                      # number of folds
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = T,                  # should class probabilities be returned
  summaryFunction=twoClassSummary  # results summary function
) 

# Hyper Parameter Tuning using tuneLength
  # Step 1: Tune hyper parameters by setting tuneLength
set.seed(100)
model_mars2 = train(Purchase ~ ., data=trainData, method='earth', tuneLength = 5, metric='ROC', trControl = fitControl)
model_mars2

  # Step 2: Predict on testData and Compute the confusion matrix
predicted2 <- predict(model_mars2, testData4)
confusionMatrix(reference = testData$Purchase, data = predicted2, mode='everything', positive='MM')

# Hyper Parameter Tuning using tuneGrid
  # Step 1: Define the tuneGrid
marsGrid <-  expand.grid(nprune = c(2, 4, 6, 8, 10), 
                         degree = c(1, 2, 3))

  # Step 2: Tune hyper parameters by setting tuneGrid
set.seed(100)
model_mars3 = train(Purchase ~ ., data=trainData, method='earth', metric='ROC', tuneGrid = marsGrid, trControl = fitControl)
model_mars3

  # Step 3: Predict on testData and Compute the confusion matrix
predicted3 <- predict(model_mars3, testData4)
confusionMatrix(reference = testData$Purchase, data = predicted3, mode='everything', positive='MM')

# eval performance of multiplie machine learning algorithms
  # training adaboost
set.seed(100)

# Train the model using adaboost
model_adaboost = train(Purchase ~ ., data=trainData, method='adaboost', tuneLength=2, trControl = fitControl)
model_adaboost

# Training Random Forest
set.seed(100)

  # Train the model using rf
model_rf = train(Purchase ~ ., data=trainData, method='rf', tuneLength=5, trControl = fitControl)
model_rf

# Training xgBoost Dart - takes a long time
set.seed(100)

  # Train the model using MARS
model_xgbDART = train(Purchase ~ ., data=trainData, method='xgbDART', tuneLength=5, trControl = fitControl, verbose=F)
model_xgbDART

# Training SVM
set.seed(100)

  # Train the model using MARS
model_svmRadial = train(Purchase ~ ., data=trainData, method='svmRadial', tuneLength=15, trControl = fitControl)
model_svmRadial

###  Run resamples() to compare the models ###
# Compare model performances using resample()
models_compare <- resamples(list(ADABOOST=model_adaboost, RF=model_rf, XGBDART=model_xgbDART, MARS=model_mars3, SVM=model_svmRadial))

# Summary of the models performances
summary(models_compare)

# Draw box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)

  # high ROC = best performing model overall
  # model that predicts positives better? look at high sensitivity

### ensembling predictions ###
# Ensemble predictions from multiple models using caretEnsemble
# we have predictions from multiple individual models. To do this we had to run the train() function once for each model, store the models and pass it to the res
# put the names of all the algorithms you want to run in a vector and pass it to caretEnsemble::caretList() instead of caret::train()
library(caretEnsemble)

# Stacking Algorithms - Run multiple algos in one call.
trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

algorithmList <- c('rf', 'adaboost', 'earth', 'xgbDART', 'svmRadial')

set.seed(100)
models <- caretList(Purchase ~ ., data=trainData, trControl=trainControl, methodList=algorithmList) 
results <- resamples(models)
summary(results)

# Box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

### combining predictions from multiple models to form final prediction ###
# combine these predicted values from multiple models somehow and make a new ensemble that predicts better
# Create the trainControl (needs to be diff from the one used to build models)
set.seed(101)
stackControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

# Ensemble the predictions of `models` to form a new combined prediction based on glm
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

# Predict on testData
stack_predicteds <- predict(stack.glm, newdata=testData4)
head(stack_predicteds)
summary(stack_predicteds)
