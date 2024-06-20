---
  title: "Salary Prediction Using R"
author: "Jerome Joshua"
date: "4/25/2020"
output:
  html_document: default
pdf_document: default
---
  
  
  #  PART 1 -Defining  the problem
  
  ### The goal of the project is to analyze the data about the various jobs of various companies and able to predict salaries of any future jobs
  
  ### Importing packages  the plot.
  

library(dplyr)
library(e1071)
library(caret)
library(ggplot2)
library(tidyr)
library(readr)
library(rattle)
library(ggpubr)
library(rpart)
library(xgboost)
library(xlsx)

options(scipen=999)
# PART 2- Discover 

### Load the input train and test files  and store as dataframe
train_features <- read_csv("C:/Users/jjosh/Desktop/train_features.csv")

train_salaries <- read_csv("C:/Users/jjosh/Desktop/train_salaries.csv")

test_features <- read_csv("C:/Users/jjosh/Desktop/test_features.csv")

### Check for duplicate values for unique key (JobId)
### There are no duplicate records  of JobId 


chk_duplicate<-train_features %>% group_by(jobId) %>% filter(n()>1)
chk_duplciate2<-train_features %>% group_by(jobId)  %>% summarize(n=n())

### Exploratory data analysis

# Below is the summary of the train_feature dataset
# Checking for how the data is structured , years of experience ranges from 0 to 24 years 
# Miles from metro city ranges from 0 to 99 miles

summary(train_features)

# Below is the summary of the train_salaries dataset
# Salary ranges from 0 k to 301 k

summary(train_salaries)

### Removing outliers
# Salary cannot be zero hence removing records where salary is 0k
# 
# train_salaries<-train_salaries[train_salaries$salary != 0, ]


### Converting the character type format varaibles into categorical variables

train_features$jobType<-as.factor(train_features$jobType)
train_features$degree<-as.factor(train_features$degree)
train_features$major<-as.factor(train_features$major)
train_features$industry<-as.factor(train_features$industry)


test_features$jobType<-as.factor(test_features$jobType)
test_features$degree<-as.factor(test_features$degree)
test_features$major<-as.factor(test_features$major)
test_features$industry<-as.factor(test_features$industry)

### Analysis of target variable 
# Histogram of the  target variable - Salary
# Target variable follows the normal distribution 

hist(train_salaries$salary,
     main="Frequency analysis of target variable-Salary",
     xlab="Salary",
     ylab="Frequency density",
     xlim=c(0,301),
     col="orange",
     freq=FALSE
)


### Analysis of 'years of experience' variable 
# No outliers in the years of experience

boxplot(train_features$yearsExperience,
        main = "Box plot analysis of 'Years of Experience'",
        xlab = "No.of years",
        col = "light blue",
        border = "brown",
        horizontal = TRUE,
        notch = TRUE
)
```

### Analysis of 'miles from metroplois' variable 
# No outliers in the  distance miles of metropolis variable.

boxplot(train_features$milesFromMetropolis,
        main = "Box plot analysis of 'Miles from metropolis",
        ylab = "No.of years",
        col = "orange",
        border = "brown",
        notch = TRUE)

### Frequency analysis of total jobs by Major
# 'None' category holds more than 50% of the total jobs
table1<-table(train_features$major)
par(oma=c(1,2,1,1))
par(mar=c(4,5,2,1))
barplot(table1[order(table1)],
        horiz = TRUE, 
        las = 1,
        col=c("beige","blanchedalmond","bisque1","bisque2","bisque3"),
        main = "Count of jobs by major wise",  xlab = "Total jobs"
)

### Frequency analysis of total jobs by Industry
# The total jobs are distributed almost equally among industries.
table2<-table(train_features$industry)
par(oma=c(1,3,1,1))
par(mar=c(4,5,2,1))
barplot(table2[order(table2)],
        horiz = TRUE, 
        las = 1,
        col=c("beige","blanchedalmond","bisque1","bisque2","bisque3"),
        main = "Count of jobs by industry wise",  xlab = "Total jobs"
)

### Frequency analysis of total jobs by degree
# High school and None categories have more jobs than other degrees.
table3<-table(train_features$degree)
par(oma=c(1,2,1,1))
par(mar=c(4,5,2,1))
barplot(table3[order(table3)],
        horiz = TRUE, 
        las = 1,
        col=c("beige","blanchedalmond","bisque1","bisque2","bisque3"),
        main = "Count of jobs by degree wise",  xlab = "Total jobs"
)

### Merging dependent and independent variables
train_all<-merge(train_features,train_salaries)

### Correlation analysis of continous variables
# Correlation  between experience and salary is only 37% which is weakly correlated.
cor.test(train_all$yearsExperience, train_all$salary, 
         method = "pearson")

# Correlation  between Miles from metropolis  and salary is only  negative 29% which is weakly correlated.
cor.test(train_all$milesFromMetropolis, train_all$salary, 
         method = "pearson")


# Correlation matirx for the continous variables 
# There is no strong relation of continous variables with target variable
M<-train_all[,c(7,8,9)]
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

M$yearsExperience<-normalize(train_all$yearsExperience)
M$salary<-normalize(train_all$salary)
M$milesFromMetropolis<-normalize(train_all$milesFromMetropolis)
M<-as.matrix(M)

library(corrplot)
Mat <- cor(M)
corrplot(Mat, method = "circle")

### Correlation analysis of categorical variables
# Since correlation cannot be computed directly , we use Analysis of variance method and finding the relation between 
# target and caterogical variables.
# 
# Relation between major and salary
boxplot(train_all$salary~train_all$major,
        main = "Box plot analysis of 'Salary vs Major'",
        xlab = "Major",
        ylab="Salary",
        col = "orange",
        border = "black",
        notch = TRUE)

aov1<-aov(train_all$salary~train_all$major)
summary(aov1)

# p-value is very less and significant.
# There is no significant difference between the categories of major with respect to salary


# Relation between degree  and salary
boxplot(train_all$salary~train_all$degree,
        main = "Box plot analysis of 'Salary vs Degree'",
        xlab = "Degree",
        ylab="Salary",
        col = "orange",
        border = "black",
        notch = TRUE)

aov2<-aov(train_all$salary~train_all$degree)
summary(aov2)

# p-value is very less and significant.
# There is  correlation   between the categories of degree  with respect to salary
# 
# Relation between industry  and salary

boxplot(train_all$salary~train_all$industry,
        main = "Box plot analysis of 'Salary vs Industry'",
        xlab = "Industry",
        ylab="Salary",
        col = "orange",
        border = "black",
        notch = TRUE)

aov3<-aov(train_all$salary~train_all$industry)
summary(aov3)

# p-value is very less and significant.
# There is  correlation   between the categories of industry with respect to salary

### Baseline model
##### Baseline model is created by using average salary of industry 

baseline<-as.data.frame(train_all%>%group_by(industry)%>%summarise(Avg.salary=mean(salary)))

train_all_with_baseline<-merge(train_all,baseline,by.x = "industry",by.y = "industry")

train_all_with_baseline$Square_error<-(train_all_with_baseline$salary-train_all_with_baseline$Avg.salary)^2

mean(train_all_with_baseline$Square_error)


# Mean square error for predicted salary (baseline) is 1367.123
# 
# Since the target variable is continous variable , we will use regression models to predict the salary 
### Goal to build 3 models and compare the better performing models

# Model 1 using Linear regression. 

model1_linear_regression<-lm (formula =salary ~	degree+major+industry+yearsExperience+milesFromMetropolis+jobType , data=train_all)
summary(model1_linear_regression)
train_all$model1<-predict (model1_linear_regression ,train_all)

train_all$Square_error<-(train_all$salary-train_all$model1)^2

mean(train_all$Square_error)

Mean square error for predicted salary using Linear regression is  384

# Model 2 using decision trees 


model2_decisiontree=rpart(salary~degree+major+industry+yearsExperience+milesFromMetropolis+jobType,data=train_all)

train_all$model2_decisiontree<-predict(model2_decisiontree ,train_all)

train_all$Square_error2<-(train_all$salary-train_all$model2_decisiontree)^2

mean(train_all$Square_error2)

Mean square error for predicted salary using Decision trees  is  710


# Model 3 using xgboost
# Data has to prepared in the format of matrix  for xgboost
# COnverting categorical features into binary varaibles using one hot encoding

category_vars = c('jobType', 'degree', 'major','industry')
dummy_vars <- dummyVars(~ jobType +  degree + major+industry, data = train_all)
one_hot_encode_category_vars <- as.data.frame(predict(dummy_vars, newdata = train_all))


# Combining category features and numeric features

xg_model_train_data<-cbind(one_hot_encode_category_vars,train_all[,c(7,8,9)])
y<-as.numeric(xg_model_train_data$salary)

Building XG boost Model 

xgb <- xgboost(data = data.matrix(xg_model_train_data[,-32]), 
               label = y,
               booster = "gblinear", 
               objective = "reg:squarederror", 
               max.depth = 5, 
               nround = 50, 
               lambda = 0, 
               lambda_bias = 0, 
               alpha = 0
)

train_all$xgboost_model = predict(xgb, newdata = as.matrix(xg_model_train_data[,-32]))

train_all$Square_error3<-(train_all$salary-train_all$xgboost_model)^2

mean(train_all$Square_error3)


# Feature Engineering 
# Mean ,median ,max and min yearsExperience are computed for each of the category variables( Industry , major, degree,JobType)

train_all<-merge(train_features,train_salaries)

jobType_stats<-train_all%>%group_by(jobType)%>%summarise(avg.salary_jobtype=mean(yearsExperience),median.salary_jobtype=median(yearsExperience),min.salary_jobtype=min(yearsExperience),max.salary_jobtype=max(yearsExperience))
industry_stats<-train_all%>%group_by(industry)%>%summarise(avg.salary_industry=mean(yearsExperience),median.salary_industry=median(yearsExperience),min.salary_industry=min(yearsExperience),max.salary_industry=max(yearsExperience))
degree_stats<-train_all%>%group_by(degree)%>%summarise(avg.salary_degree=mean(yearsExperience),median.salary_degree=median(yearsExperience),min.salary_degree=min(yearsExperience),max.salary_degree=max(yearsExperience))
major_stats<-train_all%>%group_by(major)%>%summarise(avg.salary_major=mean(yearsExperience),median.salary_major=median(yearsExperience),min.salary_major=min(yearsExperience),max.salary_major=max(yearsExperience))


mean_jobtype<-merge(jobType_stats,train_all[,c(1,3)])
mean2_jobtype<-merge(industry_stats,train_all[,c(1,6)])
mean3_jobtype<-merge(degree_stats,train_all[,c(1,4)])
mean4_jobtype<-merge(major_stats,train_all[,c(1,5)])


train_all_new<-merge(train_all,mean_jobtype,by="jobId")
train_all_new<-merge(train_all_new,mean2_jobtype,by="jobId")
train_all_new<-merge(train_all_new,mean3_jobtype,by="jobId")
train_all_new<-merge(train_all_new,mean4_jobtype,by="jobId")
final_data_<-train_all_new[,c(-1,-2,-10,-15,-20,-25)]
names(final_data_)[1]<-"jobType"
names(final_data_)[2]<-"degree"
names(final_data_)[3]<-"major"
names(final_data_)[4]<-"industry"

dummy_vars <- dummyVars(~ jobType +  degree + major+industry, data = train_all)
one_hot_encode_category_vars <- as.data.frame(predict(dummy_vars, newdata = final_data_))
xg_model_train_data<-cbind(one_hot_encode_category_vars,final_data_[,c(-1,-2,-3,-4)])
names(xg_model_train_data) <- gsub("[.]", "", names(xg_model_train_data)) 
y<-as.numeric(xg_model_train_data$salary)

# Linear regression model after feature engineering
# The MSE for linear regression model after feature engineering is  384

model1_linear_regression<-lm (formula =salary ~	. , data=final_data_)
summary(model1_linear_regression)
train_all$model1<-predict (model1_linear_regression ,final_data_)

train_all$Square_error<-(train_all$salary-train_all$model1)^2

mean(train_all$Square_error)
plot(model1_linear_regression$fitted.values,model1_linear_regression$residuals)

#From the plot there exists heteroskedasticity ie) larger differences occur when response value is larger

# Decision tree model after feature engineering 
#The MSE for decision tree model after feature engineering is  701

model2_decisiontree=rpart(salary~.,data=final_data_)

train_all$model2_decisiontree<-predict(model2_decisiontree ,final_data_)

train_all$Square_error2<-(train_all$salary-train_all$model2_decisiontree)^2

mean(train_all$Square_error2)
fancyRpartPlot(model2_decisiontree)


# XG boost model after feature engineering
# Combining category features and numeric features
# The MSE of  xgboost  model after feature engineering is  103
 
# Building XG boost Model 

xg_model_train_data[,c(1:48)] <-round(xg_model_train_data[,c(1:48)],0)
xgb_final <-xgboost(data = data.matrix(xg_model_train_data[,-32]), 
                    label = y,
                    booster = "gblinear", 
                    objective = "reg:squarederror", 
                    max.depth = 5, 
                    nround = 50, 
                    lambda = 0, 
                    lambda_bias = 0, 
                    alpha = 0
)

xg_model_train_data$xgboost_model = predict(xgb_final, newdata = as.matrix(xg_model_train_data[,-32]))

xg_model_train_data$Square_error3<-(xg_model_train_data$salary-xg_model_train_data$xgboost_model)^2

mean(xg_model_train_data$Square_error3)

### Plot of important variables usign XG boost model

importance_matrix <- xgb.importance( model = xgb_final)
xgb.plot.importance(importance_matrix,top_n = 10)



# Selection of best model 

MSE of linear regression model is 384
MSE of Decisin tree regression is 701
MSE of XG boost model regression is 103.

The best model is XG boost model.


# PART 4 -Deployment

### Scoring the test data using the final XG boost model.
#Importing the test data and extracting features and scoring the test data using the final model

test_features <- read_csv("C:/Users/jjosh/Desktop/test_features.csv")

jobType_stats<-test_features%>%group_by(jobType)%>%summarise(avg.salary_jobtype=mean(yearsExperience),median.salary_jobtype=median(yearsExperience),min.salary_jobtype=min(yearsExperience),max.salary_jobtype=max(yearsExperience))

industry_stats<-test_features%>%group_by(industry)%>%summarise(avg.salary_industry=mean(yearsExperience),median.salary_industry=median(yearsExperience),min.salary_industry=min(yearsExperience),max.salary_industry=max(yearsExperience))

degree_stats<-test_features%>%group_by(degree)%>%summarise(avg.salary_degree=mean(yearsExperience),median.salary_degree=median(yearsExperience),min.salary_degree=min(yearsExperience),max.salary_degree=max(yearsExperience))

major_stats<-test_features%>%group_by(major)%>%summarise(avg.salary_major=mean(yearsExperience),median.salary_major=median(yearsExperience),min.salary_major=min(yearsExperience),max.salary_major=max(yearsExperience))


mean_jobtype<-merge(jobType_stats,test_features[,c(1,3)])
mean2_jobtype<-merge(industry_stats,test_features[,c(1,6)])
mean3_jobtype<-merge(degree_stats,test_features[,c(1,4)])
mean4_jobtype<-merge(major_stats,test_features[,c(1,5)])


train_all_new<-merge(test_features,mean_jobtype,by="jobId")
train_all_new<-merge(train_all_new,mean2_jobtype,by="jobId")
train_all_new<-merge(train_all_new,mean3_jobtype,by="jobId")
train_all_new<-merge(train_all_new,mean4_jobtype,by="jobId")
final_data_<-train_all_new[,c(-1:-6)]
names(final_data_)[3]<-"jobType"
names(final_data_)[8]<-"industry"
names(final_data_)[13]<-"degree"
names(final_data_)[18]<-"major"
dummy_vars <- dummyVars(~ jobType +  degree + major+industry, data = final_data_)
one_hot_encode_category_vars <- as.data.frame(predict(dummy_vars, newdata = final_data_))
test_data_scored<-cbind(one_hot_encode_category_vars,final_data_[,c(-3,-8,-13,-18)])
names(test_data_scored) <- gsub("[.]", "", names(test_data_scored)) 


final_data_$predicted_salary<- predict(xgb_final, newdata = as.matrix(test_data_scored))
write.xlsx(final_data_,"test_scored.xlsx")
