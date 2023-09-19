# Data Roles Salary Prediction: Project Overview:

* Developed a model that estimates data roles(data scientists, data analysts, data engineers, ml engineer) salaries (MAE ~ $ 18K) to help them better negotiate their salary when they apply for a job.
* Scraped over 1000 job descriptions from glassdoor using python and selenium
* Applied full Data Science Pipeline: Data Exploring, Data preprocessing, Exploratory Data Analysis, Model Building.
* Using Feature Engineering, I was able to extract some new features from every job description (tools used, technlogies, education) to quantify the value companies put on these features.
* Optimized Lasso, Support vector machines and Random Forest Regressors using GridsearchCV to reach the best model. 
* Utilized the power of Ensemble methods using a voting classifier on my top 3 models.
* Built a client communicating API using fast api

## Code and Resources Used 
**Python Version:** 3.11.4  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, selenium, flask, json, pickle  
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  

## Web Scraping
scraped over 1000 unique job postings from glassdoor.com. With each job, we got the following:
*	Job title
*	Salary Estimate
*	Job Description
*	Rating
*	Company 
*	Location 
*	Company Size
*	Company Founded Date
*	Type of Ownership 
*	Industry
*	Sector
*	Revenue

## Data Preprocessing
After scraping the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:

*	Removed rows without salary 
*	Parsed numeric data out of salary column and created average salary column
*	Extracted employer provided salary and hourly wages columns
*   Simplified job titles in terms of titles and seniority
*   Created Description length of every job description
*   Featured tools, Technologies, education columns from job descriptions
*	Removed the tailing rating out of company text  
*	Transformed founded date into age of company 
*   Dealt with some of the missing data in remaining categorical columns

## Exploratory Data Analysis
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights of some significant data visualizations. 

![alt text](/Notebooks/images/EDA/corr_matrix.png "Numerical attributes correlation")
* Both desc_len, tools, techs and employer_provided columns has a slightly positive correlation with avg_salary column
* except hourly column has a slightly negative correlation with avg_salary column

![alt text](/Notebooks/images/EDA/roles_avg_salary.png "Data Roles with their avg_salary")
* looks like Machine Learning Engineer are the most paid and Data Analysts are least paid.

![alt text](/Notebooks/images/EDA/state_avg_salary.png "States with avg_salary")
* Seems like California and Newyork are the highest paying states, maybe because they have the headquarters of big tech companies

![alt text](/Notebooks/images/EDA/job_title%20histogram_plot.png "Job titles histogram")
* Seems like the market needs more Data Engineers than any other Data Roles

![alt text](/Notebooks/images/EDA/job_state%20histogram_plot.png "Job state histogram")
* There are alot of opputunities in Remote jobs, also Jobs needed in state of California.

## Model Building 

First, I Selected MAE as a Performance measure, Since we are going to treat all errors equally and we may have alot of outliers

Then, I splitted the dataset Using Stratified Sampling to make sure that test is representative of the whole dataset since the dataset is small

After that, I applied the transformation pipelines by transforming categorical variables into dummy variables, build two pipelines one for Support vector machine using median imputer and MinMaxScaler and another pipeline for other regressors using median imputer and Standard Scaler


I tried three different models: Lasso Regression, Random Forest Regressor and Support Vector Machines and evaluated them using Cross-validation.  

I tried three different models:
*	**Lasso Regression** – Because of the ability of lasso regression to eliminate the weights of least important features cause the sparsity of our categorical variables.
*	**Random Forest** – utilizing the power of ensemble learning and the randomness which trades a higher bias for a lower variance with the sparsity associated with our data.
*   **Support Vector Machines** - apowerful and versatile ml model that are particularly well suited for complex small or medium sized datasets

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Random Forest** : MAE = 19.9
*	**Lasso Regression**: MAE = 20.05
*	**SVM Regression**: MAE = 18.04

## Productionization 
In this step, I built a fast API endpoint that was hosted on a local webserver. The API endpoint takes in a request with a list of values from a job listing and returns an estimated salary. 
