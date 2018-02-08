#Loading Neccessary libraries

library(caret)
library(kernlab)
library(dplyr)
library(readr)
library(ggplot2)
library(gridExtra)


#Loading Data
train_full <- read.csv("mnist_train.csv", header = F)
test_full <- read.csv("mnist_test.csv",header = F)

#Data Understanding,Preperation and EDA
#Understanding Dimensions
dim(train_full)
dim(test_full)

#Structure of the dataset
str(train_full)
str(test_full)

#printing first few rows
head(train_full)
head(test_full)

#Exploring the data
summary(train_full)
summary(test_full)

#checking missing value
sum(sapply(train_full, function(x) sum(is.na(x))))
sum(sapply(test_full, function(x) sum(is.na(x))))
#no missing values hence no missing value treatment required

#Model building and evaluvation----
#taking 2% data of the train data due to PC performance/Time constraints
set.seed(1)
train.indices = sample(1:nrow(train_full), 0.02*nrow(train_full))
train <- train_full[train.indices, ]

#Making our target class to factor
train$V1 <-factor(train$V1)

#Constructing Model

#Using Linear Kernel
Model_linear <- ksvm(V1~ ., data = train, scale = F, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, test_full)
#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,test_full$V1)
#Accuracy 89% sensitivity to several classes are below 88%

#Using poly Kernel
Model_poly <- ksvm(V1~ ., data = train, scale = F, kernel = "polydot")
Eval_poly <- predict(Model_poly, test_full)
#confusion matrix - poly Kernel
#Accuracy 89%
confusionMatrix(Eval_poly,test_full$V1)

#Using RBF Kernel
Model_RBF <- ksvm(V1~ ., data = train, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, test_full)
#accuracy increased by using RBF
#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,test_full$V1)


#Using hyperbolic Kernel
Model_h <- ksvm(V1~ ., data = train, scale = F, kernel = "tanhdot")
Eval_h <- predict(Model_h, test_full)
#confusion matrix - poly Kernel
confusionMatrix(Eval_h,test_full$V1)

#Using RBF Kernel
Model_RBF <- ksvm(V1~ ., data = train, scale = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, test_full)
#confusion matrix - poly Kernel
#Accuracy has decreased considerably
confusionMatrix(Eval_RBF,test_full$V1)

#Can go for RBF or Poly for fine tuning 

############   Hyperparameter tuning and Cross Validation #####################

trainControl <- trainControl(method="cv", number=5)

metric <- "Accuracy"


set.seed(7)
grid <- expand.grid(.sigma=seq(6,9,by = 1), .C=seq(5,20,by = 5) )


fit.svm <- train(V1~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)
plot(fit.svm)
#very low Accuracy


#decreasing Sigma increasing C
set.seed(7)
grid <- expand.grid(.sigma=seq(0.01,0.05,by = 0.01), .C=seq(5,20,by = 5) )


fit.svm <- train(V1~., data=train, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)
plot(fit.svm)
Eval_radial<- predict(fit.svm, test_full)
confusionMatrix(Eval_radial,test_full$V1)

#trying poly instead 
set.seed(7)
grid_poly <- expand.grid(degree = 1:2,C=c(0.1,0.5,1,2),scale = c(.5,.1,0.01) )

fit.svm <- train(V1~., data=train, method="svmPoly", metric=metric, 
                 tuneGrid=grid_poly, trControl=trainControl)

print(fit.svm)
plot(fit.svm)
Eval_poly<- predict(fit.svm, test_full)
confusionMatrix(Eval_poly,test_full$V1)
#The final values used for the model were degree = 2, scale = 0.01 and C = 0.1.

#trying poly with increased C and degree of polynomial
set.seed(7)
grid_poly <- expand.grid(degree = 3:4,C=c(4,6,8,10),scale = c(.5,.1,0.01) )

fit.svm <- train(V1~., data=train, method="svmPoly", metric=metric, 
                 tuneGrid=grid_poly, trControl=trainControl)

print(fit.svm)
plot(fit.svm)
#Accuracy has decreased hence there is no need to increase the degree of polynomial
Eval_poly<- predict(fit.svm, test_full)
confusionMatrix(Eval_poly,test_full$V1)

#Finally we choose poly with degree =2 as better option than compared to radial or RBF
#The final values used for the model were degree = 2, scale = 0.01 and C = 0.1
grid_poly <- expand.grid(degree = 2,C=0.1,scale =0.01 )
Final_model <- train(V1~., data=train, method="svmPoly", metric=metric, 
                     tuneGrid=grid_poly, trControl=trainControl)
Eval_poly<- predict(Final_model, test_full)
confusionMatrix(Eval_poly,test_full$V1)
#with Accuracy : 0.918, with both sensitivity and specificity above 88% for all the classes
