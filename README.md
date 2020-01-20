# capstone-project

# What makes Pop music POP
Using machine learning to predict what makes a Spotify song popular 

## Table Of Contents
* [Technologies Used](https://github.com/georgemccrae/capstone-project/blob/master/README.md#technologies-used)
* [Packages Used](https://github.com/danch12/GA_Capstone/blob/master/README.md#packages-used)
* [Methods Used](https://github.com/danch12/GA_Capstone/blob/master/README.md#methods-used)
* [Introduction](https://github.com/danch12/GA_Capstone/blob/master/README.md#introduction)
* [Gathering the Data](https://github.com/danch12/GA_Capstone/blob/master/README.md#gathering-the-data)
* [Cleaning the Data and Feature Engineering](https://github.com/danch12/GA_Capstone#cleaning-the-data-and-feature-engineering)
* [EDA](https://github.com/danch12/GA_Capstone#eda)
* [Modelling](https://github.com/danch12/GA_Capstone#modelling)
* [NLP and Further Modelling](https://github.com/danch12/GA_Capstone#nlp-and-modelling)
* [Stacking Models](https://github.com/danch12/GA_Capstone#stacking-models)
* [Evaluation](https://github.com/danch12/GA_Capstone#evaluation)
* [Final Thoughts and Plans for the Future](https://github.com/danch12/GA_Capstone#final-thoughts-and-plans-for-the-future)

## Technologies Used
* Python 3.0
* Jupyter Notebook

## Packages Used
* Pandas
* Scikit-learn
* Matplotlib
* Numpy

## Methods Used

* Web Scraping
* Data Visualization
* Regression


## Introduction
Looking across the internet I saw that Lending Club (a peer to peer lending company) releases a massive amount of data every quarter on the status of every loan opened with them during that quarter. Therefore I thought it would be really interesting to predict the outcome of loans based on information you would know during the lifetime of the loan.The results of this project could have an exciting impact on how Lending Club looks at it's loanees as if we can predict a loan is going to default then we can take preventative measures to stop the loan from defaulting. Additionally as predicting the outcome of loans is quite difficult, it required me to seek out niche, powerful techniques that were previously unknown to me.


## Gathering the Data


* [Link to beginning of relevant section of Notebook](https://github.com/danch12/GA_Capstone/blob/16b422104066f7b96929d8ae142c9320008343b4/Data%20Gathering%20and%20Cleaning%20Stage.ipynb#L53)

First I used the Billboard API to acquire information about the top 100 songs since 1958; including the week ID, chart position. There was extensive cleaning to remove featuring artists. I then used Spotify’s API to extract information about the artist (number of followers, genre) and the musical components (e.g. danceability, tempo, duration) of the respective tracks. 

* [Scraping data from Wikipedia](https://github.com/danch12/GA_Capstone/blob/master/Data%20Gathering%20and%20Cleaning%20Stage.ipynb?short_path=8abe515#L258)

* [Getting US census data](https://github.com/danch12/GA_Capstone/blob/master/Data%20Gathering%20and%20Cleaning%20Stage.ipynb?short_path=8abe515#L590)

For this project most of the data was obtained from the Lending Club website, they provide all of their loan data in some handy csv files. To supplement this, I also included data scraped from Wikipedia  and the US census to get a better idea of the loanee's income status compared to their state average on the intuition that money goes further in different parts of the US. For example in San Francisco 100K may not be so much compared to somewhere like Alabama.

Sources of the data -
* [Lending Club data](https://www.lendingclub.com/info/download-data.action)
* [Wikepidia average income data](https://en.wikipedia.org/wiki/Household_income_in_the_United_States)
* [US Census data](https://data.census.gov/cedsci/table?q=median%20income&g=&hidePreview=true&table=S1901&tid=ACSST1Y2018.S1901&t=Income%20%28Households,%20Families,%20Individuals%29&lastDisplayedRow=16&vintage=2018&mode=)


## Cleaning the Data and Feature Engineering

I then removed outliers before performing some exploratory data analysis. My initial visual analysis graphically illustrated trends in the musical components, but more importantly to illustrate that my target variable, Spotify popularity, was  skewed to be higher the more recent it is. Importantly, I engineered some new features: ‘time since release’, ‘artist familiarity’ and ‘artist longevity’ to significantly improve the predictive power of my model.


* [Link to beginning of relevant section of Notebook](https://github.com/danch12/GA_Capstone/blob/master/Data%20Gathering%20and%20Cleaning%20Stage.ipynb?short_path=8abe515#L816)


Even though the data came straight from Lending club it was fairly messy with quite a lot of NA values. I assumed that most of these were due to attributes that did not apply to the loanee, for example if a loanee took out a loan by themselves the joint income column would be a NA value. I grouped all of the cleaning steps into one function for ease of use. This function can be seen [here](https://github.com/danch12/GA_Capstone/blob/78653880378cc5c4d6ab4b02d54b2048a1ccda0c/Data%20Gathering%20and%20Cleaning%20Stage.ipynb#L840)




Once the data was clean I created a couple of columns that made sense to me, such as the loanee's income vs the state average and how long it has been since the first credit line for the account was opened. 


At this stage I made the decision to limit my dataset to only include loans from 2014 which were given a grading of C or lower. This is because the dataset was massive and from doing some initial modelling I realised that the time it took to run any form of model would lead to me falling behind and missing deadlines.Because of this I decided to look at loans that were higher risk with higher returns for investors as those are the loans that have the greatest payoff. 


## EDA


[Link to EDA](https://github.com/danch12/GA_Capstone/blob/master/EDA.ipynb)


After cleaning the dataset I decided to explore the data visually. First I created some correlation heatmaps. You can see from below that there are a couple of areas that look extremely correlated, however a lot of these variables were later dropped before modeling because you would only know about them once the loan had completed. Having said that, the heatmap still indicates that using PCA would be a good option and this is an avenue I would like to look into in the future. After looking at the correlation of the variables generally, I looked the correlation between my target variable and my independent variables. Unfortunately almost all the most correlated variables had to be removed for the same reason as above. This left me with a couple of variables that had some correlation with loan outcome but nothing standout. The next step was to create bar graphs that visualized the distribution of good vs bad loans in different categorical variables in the data. Finally I used scatter graphs and histograms to further explore the relationship between various variables, focusing mainly on loanee income as on the face of it, income would seem like a key factor in loans defaulting.

Following on from this I looked at the distribution of good vs bad loans in different categorical variables in the data.

The main takeaways from the EDA were - 
1) We should focus on lower grade loans as they seem the most volatile
2) Income does not have as big of an impact on loans being paid as one may believe
3) There does not seem like any linear seperation between the two classes so linear models may perform badly
4) PCA seems like a good tool to use in this project as many independent variables are correlated





## Modelling 

[Link to beginning of modelling section](https://github.com/danch12/GA_Capstone/blob/master/Capstone%20Modelling%20and%20Evaluation%20phase.ipynb?short_path=90fa5f0#L262)

For my modelling I used linear regression, naive bayes and SVM to get a high score of 0.68, a huge increase from 0.22 after I engineered the new features. By examining my coefficients I found that the variables: number of Spotify followers, artist familiarity and track longevity had the largest impact on track popularity.


As alluded to in the EDA section, I was not hopeful when running a logistic regression model so I quickly diverted my attention towards other tree based models such as random forest and ada boost. I found much more success in these models so eventually I tried an XG boost model to comparitively great success with a [cv](https://github.com/danch12/GA_Capstone/blob/master/Capstone%20Modelling%20and%20Evaluation%20phase.ipynb?short_path=90fa5f0#L642) score of 0.804  . Overall I still was not happy with the results from the models which lead me to using NLP.





## NLP and Modelling

[Link to processing job titles](https://github.com/danch12/GA_Capstone/blob/master/Data%20Gathering%20and%20Cleaning%20Stage.ipynb?short_path=8abe515#L1073)

[Link to running models with job titles included](https://github.com/danch12/GA_Capstone/blob/master/Capstone%20Modelling%20and%20Evaluation%20phase.ipynb?short_path=90fa5f0#L925)

I will keep the NLP section brief as it had very little effect on the performance of any model. I used a count vectorizor on the employment title column with a high minimum appearance limit as I wanted the model to generalize well. Additionally as the models were already taking a long time to run I only included words that were discrinatory, for example 74% of engineers fully paid off their loan compared to a baseline of 66% so the word engineer was included. In the future I will improve this section of my project and expand the amount of words included. Model performance did not increase after including the job titles. 


## Stacking Models

As NLP did not provide the results I wanted, I took a different path towards stacking models. This part of my project I found incredibly interesting and would like to do more of in the future.

Before I start there are some really good articles on model stacking that helped me a lot with this part of my project-

1) First and foremost the ML Wave Ensemble guide seems to be the holy book on stacking models, it can be found [Here](https://mlwave.com/kaggle-ensembling-guide/)
2) For a slightly simpler overview on stacking models, there is a really good kaggle article using the titanic dataset [Here](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)
3) This is not an article but this script highlighted how you can reuse the same models to great effect [Here](https://github.com/emanuele/kaggle_pbr/blob/master/blend.py)
4) Finally most of the above articles were quite old so I thought I'd include one that has more up to date code [Here](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/)

Explaination of Model Stacking-

![Picture of Spaceship](https://cnet3.cbsistatic.com/img/MuWcyowyoNg76mJLO5a_7T-LB0c=/940x0/2016/11/11/4d0d9c72-97f0-41b0-965f-adb08b7e90c9/lexus-spaceship-valerian.jpg)

The ML Wave article has an great example on why you would want to stack models on top of each other, this example looks at the first spaceships and how messages were communicated.

![Communications image](https://media.npr.org/assets/img/2013/10/02/mission-control_wide-a5420ab5246dcadee1a8ae74fc6d2dcd308bab8f-s800-c85.jpg)

You could imagine that making a communication error whilst trying to land on the moon could be very costly as lives could be lost. Therefore to solve this the spacemen used [Repitition coding](https://en.wikipedia.org/wiki/Repetition_code) meaning they would send the same message a number of times and then do a majority vote.

![Repition Code](http://www.inference.org.uk/mackay/itprnn/1997/l1/img13.gif)

Now coming back to modelling, say we had 3 models all with 0.8 accuracy and predictions that are not correlated (important) we can take a majority vote and increase accuracy to almost 0.9. Now if we take this further and instead of doing a simple majority vote, we use the first layer model predictions as features for a second layer model, theoretically increasing accuracy.

However the first step in stacking models is to make sure that your first layer models are not predicting the same thing otherwise there is no use to this exercise. This is a reason why K- nearest neighbours is often used.

![Picture of prediction correlation heatmap](https://github.com/danch12/GA_Capstone/blob/master/pictures_for_readme/Screenshot%202019-12-20%20at%2012.04.43.png)

You can see that the model predictions are quite uncorrelated which means model stacking may be beneficial. For stacked model I used K- nearest neighbours, XG boost, ada boost, random forest, extra trees and logistic regression with the second layer model being another XG boost. I created a [class](https://github.com/danch12/GA_Capstone/blob/master/Capstone%20Modelling%20and%20Evaluation%20phase.ipynb?short_path=90fa5f0#L1418) that had similar functionality to an average sk learn class with a added function that allowed for getting first layer predictions and storing them within the class. This function can be seen below, but basically what it does is very similar to cross validation except instead of scoring it is making predictions on the out of fold fold. This is important as you do not want to train on the data that you make predictions on. The only other terms that may be slightly confusing are passthrough and use probability. Passthrough means that the second layer model also sees the training data as opposed to only seeing the predictions made by the first layer, in this project allowing passthrough lead to better scores but it seems uncommon to allow this to happen due to fears of overfitting. Finally use probability just means that instead of the model predicting the class for an observation it instead predicts the probability of the observation being in that class.



The results from using the stacked model were ok and were better than using the original XG boost model however I believe that more work needs to be done in fine tuning the hyper parameters for this model. Overall the stacked model outperforms the original model in almost every metric which I will go into further in the evaluation section of this readme. Interestingly although logistic regression performed badly on it's own, when used as a first layer model it becomes one of the more important first level models. [Here](https://github.com/danch12/GA_Capstone/blob/master/Capstone%20Modelling%20and%20Evaluation%20phase.ipynb?short_path=90fa5f0#L2057) is a full list of the first layer models and their feature importances.



## Evaluation

[Link to beginning of evaluation section](https://github.com/danch12/GA_Capstone/blob/master/Capstone%20Modelling%20and%20Evaluation%20phase.ipynb?short_path=90fa5f0#L2394)

To give an overview of model performance, the stacked model had a cv accuracy score of 0.815 compared to the 0.804 of the original XG boost. Further the ROC area under the curve for the stacked model was 0.9 compared to the 0.87 of the original model. Last the precision scores close but the stacked model still outperformed the unstacked model.

Finally I thought that it could be a lot more harmful to predict a loan to be payed off, for it to then default. Therefore I reduced the amount of false negatives by reducing the threshold probability at which the model predicted a loan to be bad. This resulted in a precision of 0.91 for predicting fully paid loans. 




## Final Thoughts and Plans for the Future

Overall it can be argued that on a large scale the difference in the two models would lead to a substantial increase in bad loan detection and therefore be worth the extra computational costs. Some important features common across all models made sense- those including late fees gathered throughout the loan, interest rate, and DTI. However it was suprising that credit inquiries into the loan had such a large effect- easily the largest impact feature.

In the future I would like to further this project by first increasing the scope of the data to include other years maybe eventually using PySpark to achieve this, and second try to obtain data on the full lifecyle of loans to gain a more concrete idea of the features available whilst a loan is still active as this part of the project proved difficult. I would also like to try different combinations of stacked models to see if they could provide better results and justify the extra complexity further. Using prediction probabilities has a lot of promise and I would want to further optimise the second level model that uses these. Finally although I tried using PCA to reduce model complexity,I did not have time to optimize hyperparameters for models using features processed by PCA, in the future I would like to do this as it may somewhat offset the increase in model complexity for a stacked model.
