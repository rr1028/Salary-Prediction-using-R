# Salary Prediction in R


# Summary 

The goal of the project is to predict salaries based on job descriptions. There are certain factors affecting  the prediction of salary like Education, degree ,Major , years of experience ,distance from metro city and  designation in the company.The project will help candidates to estimate their current salary and to predict their new salary if they wanted to change job .Moreover ,the project will help the companies to predcit the salaries of new hires.

# About the data

There are three files of input data : train_features, test_features, train_salaries.
The training data is split into two files namely train_features and train_salaries. Following are the features and their descriptions:

* JobId: The Id of job. It is unique for every employee.
*	companyid: The Id of company.
* JobType: The designation in the company.
* degree: Highest degree of a Employee.
* major: The field or subject on which, employee had completed his/her degree from university.
* industry: The field of company like Health, Finance, Oil etc.
* YearsofExperience: Years of Experiene of an Employee in the job.
* milesFromMetropolis: The distance in miles, the employee lives away from his/her office.
* salary: This is the target variable. It is the amount each employee receives.

# Descriptive statistics
The training data consists of 1 million records and 9 variables.

![alt text](https://github.com/jjoshua1995/Salary-Prediction-in-R/blob/master/Figures/Descriptive%20stats.PNG)

# Exploratory data analysis

The target variable is distributed normally.The value ranges from 0 to 300k.
![alt text](https://github.com/jjoshua1995/Salary-Prediction-in-R/blob/master/Figures/Histogram_target.PNG)

There are two variable which are continous variables.Below are the boxplots of milesFromMetropolis and yearsOfExperience 
showing there are not outliers in these variables.

![alt text](https://github.com/jjoshua1995/Salary-Prediction-in-R/blob/master/Figures/Boxplot_years.PNG)

Similarly , the exploratory data analysis has been conducted for all the categorical varaibles similar to the numerical variables.
Below plot shows the frequency analysis of jobs by degree wise.

![alt text](https://github.com/jjoshua1995/Salary-Prediction-in-R/blob/master/Figures/Major_histrogram.PNG)

# Correlation analysis

Correlation analysis test are conducted to see any underlying realtion exists between the variables in the data.
For numeric variables, normal correlation test is done and for categorical varaibles , analysis of variance tests are done.

Below plot is  the correlation matrix  for the numeric variables, where salary and years of experience are positive  correalted
and miles from metroplois and salary are negatively correlated.

![alt text](https://github.com/jjoshua1995/Salary-Prediction-in-R/blob/master/Figures/COrrelation%20plot.PNG)

For the categorical variables , correlation test cannot be conducted since they are not numeric variables.Hence ANOVA test is conducted
to see if the categories show any realtion to the target variable.

Below are the plots of various categories and the relation with the target variable.
Intrestingly ,all category variable thorugh ANOVA test show significant results to the target variable.

![alt text](https://github.com/jjoshua1995/Salary-Prediction-in-R/blob/master/Figures/boxplot_salary_vs_industry.PNG)

![alt text](https://github.com/jjoshua1995/Salary-Prediction-in-R/blob/master/Figures/boxplot_salary_vs_major.PNG)

Major - Engineering degree candiadtes gets highest salary and NONE category has lowest salary.

Degree- Doctoral degree candidates gets higher salaries and NONE category gets lowest salary.

Jobtype- CEO position has highest salary, whereas Janitor has lowest. The salary increase rapidly according to level of jobType.

Industry- The Finance & Oil are highest paying industries. 


# Data Preprocessing

There are some outliers in the target variable (Salary) since 0 cannot be a value in the target variable.So, outliers are removed from the training data.
Also , the features are converted to their respective data types.ie) category type variables are converted from character type to factor 
type variables.

# Feature Engineering 

Label encoding is applied to transform categorical features into numeric features. To enhance the performance of the models, following new features have been created by grouping Jobtype, degree, major, industry  based on  years of experience:

group_mean :- Average years of experience of the group.
group_median :- Median years of experience of the group.
group_max :- Maximum years of experience of the group.
group_min :- Minimum years of experience of the group.


### Evaluation Metric

To measure the effectiveness of the models that we are going to develop, we have used mean squared error (MSE) as evaluation metric.

## Baseline model

For a baseline model,it is good to consider the average of the target variable. Here, in this project we take the average salary 
per industry .

## Model 1 -- Linear Regression

First model we selected was the linear regression model.Because the target variable is continous and so a regression model is needed.
The mean MSE error obtained is :  384
Below is the plot of reisduals and expected values and we could see the difference in errors enlarge when the target value enlarges.
The funnel shaped is created because of this character of the target variable.

![alt text](https://github.com/jjoshua1995/Salary-Prediction-in-R/blob/master/Figures/Linear_regression.PNG)

## Model 2 -- Decision tree

Second model selected was decsion tree regression model. The mean MSE error obtained is 701.
Below is the decision tree with various branching conditions.

![alt text](https://github.com/jjoshua1995/Salary-Prediction-in-R/blob/master/Figures/Decisin_tree.PNG)

## Model 3 --XG Boost

Third model selected is the XG boost model. The mean MSE error obtained is 360.
Below is the plot for important features obtained through XG boost model.

![alt text](https://github.com/jjoshua1995/Salary-Prediction-in-R/blob/master/Figures/Important_features.PNG)

 
| Model             | MSE |
|-------------------|-----|
| Linear regression | 384 |
| Decision tree     | 701 |
| XG boost model    | 360 |

### Best model selection

From the three models XG boost model is performing better than the other two models.Feature engineering was one of the important factors 
that imprived the performance of the model.


# Conclusion

We can conclude that we have developed a model with MSE of 360, to predict the salary based on the features given and newly generated features.

Salary varies according to the following

* Salary decreases linearly with miles away from city
* Salary increases linearly with years of experience
* Job position: CEO > CTO, CFO > VP > Manager > Senior > Junior > Janitor
* Oil and finance industries are the highest paying sectors, while service and education are the lowest paying.

We can further try to improve the model by generating new features.
Years of experience and designation play vital roles in the predictive power of the model.





