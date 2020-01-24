# An investigation into what makes music on Spotify popular using machine learning

Using machine learning to predict what makes a Spotify song popular 

## Table Of Contents
* [Technologies Used](https://github.com/georgemccrae/capstone-project/blob/master/README.md#technologies-used)
* [Packages Used](https://github.com/georgemccrae/capstone-project/blob/master/README.md#packages-used)
* [Methods Used](https://github.com/georgemccrae/capstone-project/blob/master/README.md#methods-used)
* [Introduction](https://github.com/georgemccrae/capstone-project/blob/master/README.md#introduction)
* [Gathering & Cleaning the Data](https://github.com/georgemccrae/capstone-project/blob/master/README.md#gathering-&-cleaning-the-data)
* [EDA](https://github.com/georgemccrae/capstone-project#eda)
* [Feature Engineering](https://github.com/georgemccrae/capstone-project#feature-engineering)
* [Modelling](https://github.com/georgemccrae/capstone-project#modelling)
* [Evaluation](https://github.com/georgemccrae/capstone-project#evaluation)
* [Final Thoughts and Plans for the Future](https://github.com/georgemccrae/capstone-project#plans-for-the-future)

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

I've always loved music. I was watching this [Youtube video](https://www.youtube.com/watch?v=scbbVSeKS4I&t=8s) on what make Lil Nas X's video so popular. It desribes how he used memes, search engine optimisation, remixs (which count towards chart placement), tiktok, memes, hopping on Red Dead Redemption's cowboy theme and classifying the song as 'country' on itunes and souncloud in order to remanouvere recommendation alogrithmns rather than trying to compete with songs in today's oversaturated hip-hip genre. All of these were ingenuious tactics which helped to make his song break records for weeks at #1. 

I set out to try and uncover whether there are any attributes of a song or an artist which made it destined to be popular.


## Gathering & Cleaning the Data

* [Link to Gathering & Cleaning Data notebook](https://github.com/georgemccrae/capstone-project/blob/master/github%20-%20scraping.ipynb#L53)

For this project the data was obtained in two parts and then merged:

### Billboard Data

#### Source 
* [Billboard API](https://github.com/guoguo12/billboard-charts)
* [Information of how the ranking system is calculated](https://en.wikipedia.org/wiki/Billboard_Hot_100#Hot_100_policy_changes) 

Billboard's ranking method for the Top 100 is excellent because it stayed relevant with its ranking policy with the changing methods of discoverning and purchasing music.   

- 1958-1991: ranking determined by ratio of singles sales and airplay
- 1991: Billboard begins collecting sales data digitally (using SoundScan) for quicker and more accurate charts
- 1998: Billboard drops requirement that song must be released as a single to appear on the chart
- 2005: Digital downloads (iTunes) included
- 2012: On-demand streaming services (Spotify, Rhapsody) included
- 2013: Video views (YouTube) included

#### Gathering 

I used guoguo12's Billboard API to acquire information about the top 100 songs since 1958; including the week ID, chart position. 

#### Cleaning 
There was extensive cleaning of the Billboard data. The largest issue was that when I put the artist's name acquired from the Billboard API into the Spotify API, if the name of the artist was too long it would produce no results - therefore I had to remove featuring artists. This was simple is there was a 'featuring'. However &, comma, 'and', slash were all used synonymously. Therefore, I searched through all artists containing a comma etc and if the name appeared elsewhere then I would assume that it was a solo artist. I manually sorted through the remaining instances where a comma appeared and made some other exceptional lists. 

![Data Scraped](https://github.com/georgemccrae/capstone-project/blob/master/images/image%201%20-%20data%20scraped.jpg
)



### Spotify Data

#### Source
* [Spotify API](https://github.com/plamere/spotipy)

#### Gathering 
The next step was to get all Spotify’s musical components (e.g. danceability, tempo, duration) aswell as information about the artist (number of followers, genre) for each respective Billboard track. I used plamere's Spotify’s API to extract this and then merged it with the Billboard data. There were  cases where the Billboard and Spotify artist names didnt match up, so I created a dicitonary of such entries and used the Python package [Fuzzy Wuzzy](https://github.com/seatgeek/fuzzywuzzy) to see if they were adequately similar. 

#### Cleaning 
The Spotify Data was mostly clean, Spotify's genre classification system provided additional challenges. The streaming service categorizes artists into over 1,300 specific, and often unheard of, music genres (anybody familiar with ["zydeco"](https://en.wikipedia.org/wiki/Zydeco)?). As the genre tags can only be acquired from the artist parameter, this creates a problem. It means that the genre label is not specific to each track but to the artist as a whole; therefore the tracks of an ecclectic artist who spans several genres (one artist had 22 genre tags) are likely to be mislabelled. 

I used a two-step process to translate Spotify's genres to my own genre definition. First, I Count-Vectorised the whole column to see which terms appeared the most. I then manually sorted through them to make a list of real genres. For example, I ignored 'new' (stemming from 'new wave'). 

* 'r&b', 'rock', 'metal', 'grunge', 'punk', 'pop', 'house', 'electronic', 'trance', 'dance', 'country', 'folk', 'jazz', 'blues', 'soul', 'disco', 'funk', 'trap', 'rap', 'freestyle', 'indie', 'classical', 'ska', 'reggae', 'dancehall', 'adult standards', 'hip hop'

I then Count-Vectorised again with these are column names. I wrote a wrote a short Python script to 'vote' on which genre to place the artist in. For instance, Spotify classifies Drake as "pop rap", "indie r&b", "alternative hip hop", and "hip hop". According to our mapping system, three of those genres fall under rap/hip-hop and one under R&B. Thus, Drake goes under rap/hip-hop.

I made a seperate notebook for cleaning the genre tags [here](https://github.com/georgemccrae/capstone-project/blob/master/github%20-%20scraping.ipynb#L53)

## Feature Engineering

* [Link to Feature Engineering notebook](https://github.com/georgemccrae/capstone-project/blob/master/github%20-%20eda%20%26%20feature%20engineering.ipynb#L816)

Once the data was clean, I ran a quick linear regression to see roughly what the cross-valdiated score was; at 0.22 I realised that this project needed a lot more work.

I engineered some new features, to significantly improve the predictive power of my model: 
* ‘track longevity’ 
* ‘artist familiarity’ 
* 'peak chart position'
* ‘time since first charting’
* 'percentage of genre dominance' - this had no impact on predicting popularity

After engineering the all-important new features, my highest cross-validated score of 0.68, a huge increase.

INSERT SLIDE FROM PRESENTATION

AMAZING PLOT IN CODE AND SCREENSHOT

## EDA

[Link to EDA notebook](https://github.com/georgemccrae/capstone-project/blob/master/github%20-%20eda%20%26%20feature%20engineering.ipynb)

My inital EDA was to look at correlation between the musical components and Spotify popularity. I found that loudness was the mostly highly correlated with the target at 0.35; overall not much correlation. Further, the highest correlation amongst the musical components was not surprisingly between 'Loudness' and 'Energy' at 0.69. Correlation between my engineering features and the target was more promising. I found that 'Time Since Release', 'Numnber of Spotify Followers' and 'Artist Familiarity' all had a correlation score of over 0.5, showing that there was at least some correlation. 

Next I examined a histogram of all the musical features. I observed that most of them were not normally distributed and therefore a Power Tranformer might be needed during the modelling stage. 

Then I graphically illustrated, with a timeseries, trends in the musical components of tracks; DETAILED ANALYSIS CONTAINED IN NOTEBOOK.

An important finding was that Spotify gives higher popularity rankings for a new releases and artists that have released new music recently. Therefore my target variable, Spotify popularity, was skewed to be higher the more recent it is. This was hence why I engineered the feature 'Time Since Release' which massively increased my models' predictiveness. Here is Spotify's description of how it is calculated.

> “The popularity of the track. The value will be between 0 and 100, with **100 being the most popular.** The popularity is calculated by algorithm and is based, in the most part, on **the total number of plays the track has had and how recent those plays are.** Generally speaking, **songs that are being played a lot now will have a higher popularity** than songs that were played a lot in the past.”*

At this point I droppped the column 'spot_artist_pop' because it's derived from the spotify track popularity. 

Finally, there were some complications in removing outliers as some tracks were classified as double their BPM (Beats Per Minute).


## Modelling 

### Predicting Spotify track popularity 

[Link to Modelling notebook](https://github.com/georgemccrae/capstone-project/blob/master/modelling%202.ipynb#L262)

My final were features were:

* 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'duration', 'key', 'mode', 'time_signature', 'spot_followers', 'track_longevity', 'peak_rank', 'time'

My highest cross-validation R2 score was 0.72 by Gridsearching on a Random Forest. Here were my top feature importances:

* spot_followers	0.378432	
* track_longevity	0.209470	
* time	0.174249	0.174249
* artist_familiarity 0.035067
* duration	0.022495	
* liveness	0.019045

I also used an Elastic Net CV on a Linear Regression to get a score of R2 score 0.54. The advantage of using linear regression of course means that these coefficients are directly interpretable: 

* time	-11.411248	
* peak_rank	-5.101219	
* track_longevity	3.552898	
* spot_followers	2.793421	
* liveness	-0.667895	
* loudness	0.532260	

## Evaluation

By examining the coefficients I found that the variables: 'Time Since First Charting', 'Number of Spotify followers', 'Artist Chart Familiarity' and 'Track Chart Longevity' had the largest impact on track popularity.

In conclusion, if you are an artist aiming to get your songs into automatically curated playlists or wonder how to get higher 
rankings in Spotify popularity index make sure you release songs frequently to stay relevant in the Spotify world. Further, if an artist wants to focus on getting a high Spotify score, then they should focus their efforts on trying to gain more followers as opposed to getting views on other mediums such as YouTube. Finally, although 'Duration' and 'Loudness' seem to have a small impact on a track's popularity, it seems as if musical components are largely irrelevant in predicting what makes a song popular. 


## Plans for the Future
Due to time constraints I focused on the musical components and artist information derived from the Spotify API. However there are several other potentially brilliant predictors of popularity I'm excited to add:

#### INSTAGRAM / SOUNDCLOUD /FACEBOOK / TWITTER FOLLOWERS	
#### LYRICS (GENIUS API)
#### NO. PRODUCERS (GENIUS API)
#### ARTIST LOCATION (WIKIPEDIA API)

I also want to investigate homoegeneity in music. It seems as if the Billboard Hot 100 will continue to musically converge, give it enough time and we’ll all be listening to the same thing.


