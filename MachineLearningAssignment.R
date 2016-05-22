## Practical Machine Learning-Predicting Exercise Activitiess

library(caret)

## Load the pml data sets
setwd("C:\\Courses\\Data8b Practical Machine Learning\\Project")
training <- read.csv(file = "pml-training.csv", header = TRUE)
testing <- read.csv(file = "pml-testing.csv", header = TRUE)
nrows <- dim(training)[1]
ncols <- dim(training)[2]
dimData <- rbind(dim(training), dim(testing))
rownames(dimData) <- c("Training", "Testing")
colnames(dimData) <- c("Rows", "Columns")
dimData

# training_complete <- training[complete.cases(training),]
# training_omit <- na.omit(training)
# dim(training)
# dim(training_complete)

str(training)
summary(training[, c("user_name", "stddev_yaw_belt", "yaw_belt")])
set.seed(1234)

inTrain <- createDataPartition(y = training$classe, p = 0.7, list = FALSE)
ptraining <- training[inTrain, ]
ptest <- training[-inTrain, ]
head(ptraining)
names(ptraining)

#modelFit <- train(classe ~ ., data = training, method = "lda")
# modelLasso <- train(classe ~ ., method="rf", data=ptraining)
# plot.enet(modelLasso$finalModel, xvar="penalty", use.color=TRUE)

# trainBelt <- training[, grep("_belt|classe", names(training))]
# featurePlot(x = trainBelt[, 7:9], y = training$classe, plot = "pairs")
# trainBelt_na <- apply(trainBelt, 2, function(x) {sum(!is.na(x))})
# names(trainBelt_na)

ntrain <- length(training$classe)
names_org <- names(training)
cat("names_org: ", length(names_org), "; n: ", ntrain)
names_id_idx <- c(1:7)
names_stat1_idx <- grep("kurtosis_|skewness_|max_|min_|amplitude_", names_org)
names_stat2_idx <- grep("kurtosis_|skewness_|max_|min_|amplitude_|stddev_|avg_|var_", names_org)
names_id_stat1_idx <- c(names_id_idx, names_stat1_idx)
names_id_stat2_idx <- c(names_id_idx, names_stat2_idx)
length(names_id_idx)
length(names_stat1_idx)
length(names_stat2_idx)
length(names_id_stat1_idx)
length(names_id_stat2_idx)

# names_notsel <- names_org[names_notsel_id]
# training_na <- training[, colSums(!is.na(training)) == ntrain]
# names_na <- names(training_na)
# cat("names_na: ", length(names_na), "; n: ", ntrain)
# names_notsel_id <- grep("kurtosis_|skewness_|max_|min_|amplitude_", names_org)
# names_notsel <- names_org[names_notsel_id]
# cat("names_notsel: ", length(names_notsel), "; n: ", ntrain)
# names_nanotsel <- intersect(names_na, names_notsel)
# cat("names_nanotsel: ", length(names_nanotsel), "; n: ", ntrain)

#featurePlot(x = training[, names_sel[4:6]], y = training$classe, plot = "pairs")

names_cor1 <- names_org[-c(names_id_stat1_idx, length(names_org))]
correlationMatrix1 <- cor(training[, names_cor1], use="pairwise.complete.obs")
highlyCorrelated1 <- findCorrelation(correlationMatrix1, cutoff = 0.75)
names_hi_cor1 <- names_cor1[highlyCorrelated1]
names_hi_cor1_idx <- which(names_org %in% names_hi_cor1)
names_not_train1_idx <- c(names_id_idx, names_stat1_idx, names_hi_cor1_idx)
length(names_not_train1_idx)
# 0.50 125
# 0.75 109
# 0.90 90
# 0.99 71

names_cor2 <- names_org[-c(names_id_stat2_idx, length(names_org))]
correlationMatrix2 <- cor(training[, names_cor2], use="pairwise.complete.obs")
highlyCorrelated2 <- findCorrelation(correlationMatrix2, cutoff = 0.75)
names_hi_cor2 <- names_cor2[highlyCorrelated2]
names_hi_cor2_idx <- which(names_org %in% names_hi_cor2)
names_not_train2_idx <- c(names_id_idx, names_stat2_idx, names_hi_cor2_idx)
length(names_not_train2_idx)
# 0.50 138
# 0.75 127
# 0.90 114
# 0.99 108

names_stat_idx <- grep("kurtosis_|skewness_|max_|min_|amplitude_|stddev_|avg_|var_", names_org)
names_id_stat_idx <- c(names_id_idx, names_stat_idx)
length(names_id_idx)
length(names_stat_idx)
length(names_id_stat_idx)

names_cor <- names_org[-c(names_id_stat_idx, length(names_org))]
correlationMatrix <- cor(training[, names_cor], use="pairwise.complete.obs")
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.75)
names_hi_cor <- names_cor[highlyCorrelated]
names_hi_cor_idx <- which(names_org %in% names_hi_cor)
names_not_train_idx <- c(names_id_idx, names_stat_idx, names_hi_cor_idx)
length(names_not_train_idx)

names_train <- names_org[-c(names_not_train_idx, length(names_org))]

n_features <- length(names_org) - 1
n_id <- length(names_id_idx)
n_stat <- length(names_stat_idx)
n_hi_cor <- length(names_hi_cor_idx)
n_train <- length(names_train)
nTab <- rbind(n_features, n_id, n_stat, n_hi_cor, n_train)
rownames(nTab) <- c("Total", "Id", "Derived", "Hi Correl.", "Predictor")
colnames(nTab) <- c("Features")
nTab

# control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# results <- rfe(training[, names_sel], training[, "classe"], sizes=c(1:52), rfeControl=control)
# print(results)
# # list the chosen features
# predictors(results)
# # plot the results
# plot(results, type=c("g", "o"))

## Random forest
library(caret)
formulaRf <- as.formula(paste("classe ~ ", paste(names_train, collapse = "+")))
#modelRf <- train(formulaRf, data = ptraining, method = "rf", ntree = 200, prox = TRUE)

ptm <- proc.time()
modelRf <- train(formulaRf, data = ptraining, method = "rf", prox = FALSE)
#modelRf <- train(formulaRf, data = ptraining, method = "parRF", mtry = 50, prox = TRUE)
proc.time() -  ptm

head(getTree(modelRf$finalModel, 1, labelVar = TRUE), 40)
harP <- classCenter(ptraining[, names_train], ptraining$classe, modelRf$finalModel$prox)
harP <- as.data.frame(harP)
harP$classe <- rownames(harP)
p <- qplot(roll_belt, yaw_belt, col = classe, data = ptraining)
p + geom_point(aes(x = roll_belt, y = yaw_belt, col = classe), size = 5, shape = 4, data = ptraining)
qplot(gyros_arm_z, yaw_belt, color = classe, data = ptraining, main = "Training Data Predictions")

trainPredicts <- predict(modelRf, ptraining)
table(trainPredicts, ptraining$classe, dnn=c("Prediction", "Training Classe"))
trainConf <- confusionMatrix(trainPredicts, ptraining$classe, dnn=c("Prediction", "Training Classe"))
trainConf[['table']]
trainConf[['overall']]['Accuracy']

testPredicts <- predict(modelRf, ptest)
ptest$correctPredict <- testPredicts == ptest$classe
table(testPredicts, ptest$classe, dnn=c("Prediction", "Validation Classe"))
#qplot(roll_belt, yaw_belt, color = predRight, data = ptest, main = "Newdata Predictions")
qplot(gyros_arm_z, yaw_belt, color = correctPredict, data = ptest, main = "Testing Data Predictions")
names(confusionMatrix(testPredicts, ptest$classe))
#[1] "positive" "table"    "overall"  "byClass"  "dots" 
testConf <- confusionMatrix(testPredicts, ptest$classe, dnn=c("Prediction", "Validation Classe"))
testConf[['table']]
testConf[['overall']]['Accuracy']

accuracyTab <- rbind(trainConf[['overall']]['Accuracy'], testConf[['overall']]['Accuracy'])
rownames(accuracyTab) <- c("Training", "Validation")
accuracyTab

testPredicts <- predict(modelRf, testing)
testing$correctPredict <- testPredicts == testing$classe
testPredicts
# [1] B A B A A E D B A A B C B A E E A B B B

## Plot tree
par(mfrow=c(1, 1))
finalModel <- modelRf$finalModel
oobErrors <- finalModel$err.rate
classError <- colnames(oobErrors)
plot(finalModel, main = "Out-of-bag Error Estimates",
     col=c("black", "cyan", "orange", "blue", "red", "blue"))
legend(320, 0.20, classError,
       lty=c(1,1),
       lwd=c(2.5,2.5), 
       col=c("black", "cyan", "orange", "blue", "red", "blue"))

ntrees <- dim(oobErrors)[1]
sprintf("%.1f", 100*oobErrors[ntrees, 1])
 
# Plot importance
varImpPlot(finalModel, main="Gini Importance")

# library(rattle)
# printRandomForests(modelRf$finalModel, 5)
# treeset.randomForest(modelRf$finalModel, n=5)
# MDSplot(modelRf$finalModel)

## Plot tree
par(mfrow=c(1, 1))
plot(modelRf$finalModel, main="Classification Tree")
text(modelRf$finalModel, use.n=TRUE, all=TRUE, cex=.8)

## Prettier plots
# library(rattle)
# fancyRpartPlot(modelRf$finalModel)

rfTree <- getTree(modelRf$finalModel, k=1, labelVar=TRUE)

# Function to print tree
# http://stats.stackexchange.com/questions/2344/best-way-to-present-a-random-forest-in-a-publication

to.dendrogram <- function(dfrep,rownum=1,height.increment=0.1){
    
    if(dfrep[rownum,'status'] == -1){
        rval <- list()
        
        attr(rval,"members") <- 1
        attr(rval,"height") <- 0.0
        attr(rval,"label") <- dfrep[rownum,'prediction']
        attr(rval,"leaf") <- TRUE
        
    }else{##note the change "to.dendrogram" and not "to.dendogram"
        left <- to.dendrogram(dfrep,dfrep[rownum,'left daughter'],height.increment)
        right <- to.dendrogram(dfrep,dfrep[rownum,'right daughter'],height.increment)
        rval <- list(left,right)
        
        attr(rval,"members") <- attr(left,"members") + attr(right,"members")
        attr(rval,"height") <- max(attr(left,"height"),attr(right,"height")) + height.increment
        attr(rval,"leaf") <- FALSE
        attr(rval,"edgetext") <- dfrep[rownum,'split var']
    }
    
    class(rval) <- "dendrogram"
    
    return(rval)
}

d <- to.dendrogram(rfTree)
str(d)
plot(d, center=TRUE, leaflab='none', edgePar(t.cex=1, p.col=NA, p.lty=0))

VI_F <- importance(modelRf$finalModel)
barplot(t(VI_F/sum(VI_F)))


importanceOrder=order(-modelRf$finalModel$importance)
names=rownames(modelRf$finalModel$importance)[importanceOrder][1:4]
paste(names, collapse = ", ")

par(mfrow=c(2, 2), xpd=NA)
for (name in names)
    partialPlot(modelRf$finalModel, ptraining[, names_train], eval(name), main=name, xlab=name)

importanceOrder <- order(-finalModel$importance)
ntop = 7
names_imp <- rownames(finalModel$importance)[importanceOrder][1:ntop]
names_imp_list <- paste(
    paste(names_imp[1:ntop-1], collapse = ", "), "and",
    names_imp[ntop])
