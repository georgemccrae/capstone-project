# capstone-project

# What makes pop music POP
Using machine learning to predict what makes a Spotify song popular 

## Table Of Contents
* [Technologies Used](https://github.com/georgemccrae/capstone-project/blob/master/README.md#technologies-used)
* [Packages Used](https://github.com/georgemccrae/capstone-project/blob/master/README.md#packages-used)
* [Methods Used](https://github.com/georgemccrae/capstone-project/blob/master/README.md#methods-used)
* [Introduction](https://github.com/georgemccrae/capstone-project/blob/master/README.md#introduction)
* [Gathering the Data](https://github.com/georgemccrae/capstone-project/blob/master/README.md#gathering-the-data)
* [Cleaning the Data and Feature Engineering](https://github.com/georgemccrae/capstone-project#cleaning-the-data-and-feature-engineering)
* [EDA](https://github.com/georgemccrae/capstone-project#eda)
* [Modelling](https://github.com/georgemccrae/capstone-project#modelling)
* [Evaluation](https://github.com/georgemccrae/capstone-project#evaluation)
* [Final Thoughts and Plans for the Future](https://github.com/georgemccrae/capstone-project#final-thoughts-and-plans-for-the-future)

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


#### 

-on souncloud / itunes he listed track's genre as 'country', an ingenuious tactic in remanouvering the alogrithmn  rather than trying to compete with songs in today's oversaturated hip-hip genre. 

-hill used memes / red-dead / search engine optimisation / tiktok / remixs (count towards chat placement) / being in multiple genre
Looking across the internet I saw that Lending Club (a peer to peer lending company) releases a massive amount of data every quarter on the status of every loan opened with them during that quarter. Therefore I thought it would be really interesting to predict the outcome of loans based on information you would know during the lifetime of the loan.The results of this project could have an exciting impact on how Lending Club looks at it's loanees as if we can predict a loan is going to default then we can take preventative measures to stop the loan from defaulting. Additionally as predicting the outcome of loans is quite difficult, it required me to seek out niche, powerful techniques that were previously unknown to me.


## Gathering the Data

* [Link to beginning of relevant section of Notebook](https://github.com/danch12/GA_Capstone/blob/16b422104066f7b96929d8ae142c9320008343b4/Data%20Gathering%20and%20Cleaning%20Stage.ipynb#L53)

For this project the data was obtained in two parts. 

Firstly I used the Billboard API to acquire information about the top 100 songs since 1958; including the week ID, chart position. Because of time constraints I ended up using a csv file from guoguo12. 

* [Billboard Top 100 Data](https://github.com/guoguo12/billboard-charts)

I then used plamere's Spotify’s API to extract information about the artist (number of followers, genre) and the musical components (e.g. danceability, tempo, duration) of the respective tracks. 

* [Spotify Data](https://github.com/plamere/spotipy)


## Cleaning the Data and Feature Engineering

* [Link to beginning of relevant section of Notebook](https://github.com/danch12/GA_Capstone/blob/master/Data%20Gathering%20and%20Cleaning%20Stage.ipynb?short_path=8abe515#L816)

There was extensive cleaning of the Billboard data. In main, because in order to extract the data of the correct artist from the Spotify API, I had to remove featuring artists. 

There weren't many NA values as the Billboard data was fairly complete, however I had to make sure that the two datasets matched. 

I then removed outliers

Once the data was clean, I engineered some new features, which seemed to be important in predicitng popularity: ‘time since release’, ‘artist familiarity’ and ‘artist longevity’ to significantly improve the predictive power of my model.


## EDA

[Link to EDA](https://github.com/danch12/GA_Capstone/blob/master/EDA.ipynb)

My initial visual analysis graphically illustrated trends in the musical components, but more importantly to illustrate that my target variable, Spotify popularity, was  skewed to be higher the more recent it is. 

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


## Evaluation



To give an overview of model performance, the stacked model had a cv accuracy score of 0.815 compared to the 0.804 of the original XG boost. Further the ROC area under the curve for the stacked model was 0.9 compared to the 0.87 of the original model. Last the precision scores close but the stacked model still outperformed the unstacked model.

Finally I thought that it could be a lot more harmful to predict a loan to be payed off, for it to then default. Therefore I reduced the amount of false negatives by reducing the threshold probability at which the model predicted a loan to be bad. This resulted in a precision of 0.91 for predicting fully paid loans. 




## Final Thoughts and Plans for the Future

obvious trend is that the Billboard Hot 100 will continue to musically converge, a path that might just be the natural progression of popular culture. give it enough time and we’ll all be listening to the same thing


Overall it can be argued that on a large scale the difference in the two models would lead to a substantial increase in bad loan detection and therefore be worth the extra computational costs. Some important features common across all models made sense- those including late fees gathered throughout the loan, interest rate, and DTI. However it was suprising that credit inquiries into the loan had such a large effect- easily the largest impact feature.

In the future I would like to further this project by first increasing the scope of the data to include other years maybe eventually using PySpark to achieve this, and second try to obtain data on the full lifecyle of loans to gain a more concrete idea of the features available whilst a loan is still active as this part of the project proved difficult. I would also like to try different combinations of stacked models to see if they could provide better results and justify the extra complexity further. Using prediction probabilities has a lot of promise and I would want to further optimise the second level model that uses these. Finally although I tried using PCA to reduce model complexity,I did not have time to optimize hyperparameters for models using features processed by PCA, in the future I would like to do this as it may somewhat offset the increase in model complexity for a stacked model.
