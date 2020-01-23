# An investigation into what makes Spotify’s music popular using machine learning

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

For this project the data was obtained in two parts (Billboard Data & Spotify Data) and then merged:

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
There was extensive cleaning of the Billboard data. The largest issue was that when I put the artist's name acquired from the Billboard API into the Spotify API, if the name of the artist was too long it would produce no results - therefore I had to remove featuring artists. 

Further, if the artist had a short name then the Spotify API would often produce data for the wrong artist; so I manually iterated through them to figure out the correct one. 

The Python package  [Fuzzy Wuzzy](https://github.com/seatgeek/fuzzywuzzy) helped massively. 



### Spotify Data

#### Source
* [Spotify API](https://github.com/plamere/spotipy)

#### Gathering 
The next step was to get all Spotify’s musical components (e.g. danceability, tempo, duration) aswell as information about the artist (number of followers, genre) for each respective Billboard track. I used plamere's Spotify’s API to extract this. The Spotify API had about 96.5% of the ~21,200 songs to appear on these charts. For the remaining 753, I plugged in average numbers for each acoustic feature, just to have some placeholder data. I then put all those acoustic metadata back into the full Billboard list. 

#### Cleaning 
The Spotify Data was mostly clean, Spotify's genre classification system provided additional challenges. The streaming service categorizes artists into over 1,300 specific, and often unheard of, music genres (anybody familiar with ["zydeco"](https://en.wikipedia.org/wiki/Zydeco)?). As the genre tags can only be acquired from the artist parameter, this creates a problem. It means that the genre label is not specific to each track but to the artist as a whole; therefore the tracks of an ecclectic artist who spans several genres (one artist had 22 genre tags) are likely to be mislabelled. 

I used a two-step process to translate Spotify's genres to my own genre definition. First, I Count-Vectorised the whole column to see which terms appeared the most. I then manually sorted through them to make a list of real genres. For example, I ignored 'new' (stemming from 'new wave'). 

* 'r&b', 'rock', 'metal', 'grunge', 'punk', 'pop', 'house', 'electronic', 'trance', 'dance', 'country', 'folk', 'jazz', 'blues', 'soul', 'disco', 'funk', 'trap', 'rap', 'freestyle', 'indie', 'classical', 'ska', 'reggae', 'dancehall', 'adult standards', 'hip hop'

I then Count-Vectorised again with these are column names. I wrote a wrote a short Python script to 'vote' on which genre to place the artist in. For instance, Spotify classifies Drake as "pop rap", "indie r&b", "alternative hip hop", and "hip hop". According to our mapping system, three of those genres fall under rap/hip-hop and one under R&B. Thus, Drake goes under rap/hip-hop.

I made a seperate notebook for cleaning the genre tags here [here](https://github.com/georgemccrae/capstone-project/blob/master/github%20-%20scraping.ipynb#L53)


## EDA

[Link to EDA notebook](https://github.com/georgemccrae/capstone-project/blob/master/github%20-%20eda%20%26%20feature%20engineering.ipynb)

My initial visual analysis graphically illustrated trends in the musical components. 

An important finding was that Spotify gives higher popularity rankings for a new releases and artists that have released new music recently. Therefore my target variable, Spotify popularity, was skewed to be higher the more recent it is. Here is Spotify's description of how it is calculated.

> “The popularity of the track. The value will be between 0 and 100, with **100 being the most popular.** The popularity is calculated by algorithm and is based, in the most part, on **the total number of plays the track has had and how recent those plays are.** Generally speaking, **songs that are being played a lot now will have a higher popularity** than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity. Note that the popularity value may lag actual popularity by a few days: the value is not updated in real time.”*

There were some complications in removing outliers as some tracks were classified as double their BPM (Beats Per Minute). There was some unavoivable subjectivity in deducing which were outliers. 

The main takeaways from the EDA were - 
1) Rock and soul were the most popular music genres from the mid-60s to mid-70s. But as soul peaked in 1974 and slowly began to fade, rock continued to climb. Its run from 1982-86, when rock musicians occupied nearly 60% of available Hot 100 spots, is by far the most dominant stretch for any one genre.

2) Despite all the attention paid to boy bands in the late '90s, it seems like R&B had no problem flourishing. Acts like Boyz II Men and Janet Jackson propelled the genre's popularity and ingratiated it with the masses.

3) Country has had a tumultuous ride in the history of popular American music. It enjoyed middling popularity through the mid-'80s, when it all but dropped off the charts. Since 1999, however, it's seen a noticeable resurgence.

4) Music trends have swung in favor of pop and hip-hop in the 2010s. Pop has owned the largest share of Billboard spots dating back to 2006, but has seen its popularity decline slightly since 2011. Meanwhile, rap has come on strong in the last two years. In fact, rap is on pace to occupy more than 30% of Hot 100 spots this year, higher than its previous peak in 2004.



## Feature Engineering

* [Link to Feature Engineering notebook](https://github.com/georgemccrae/capstone-project/blob/master/github%20-%20eda%20%26%20feature%20engineering.ipynb#L816)


Once the data was clean, I engineered some new features, which seemed to be important in predicitng popularity: ‘time since release’, ‘artist familiarity’ and ‘artist longevity’ to significantly improve the predictive power of my model.

# genre classifcation method is not accurate, genre had no impact on the pop model




## Modelling 

[Link to Modelling notebook](https://github.com/georgemccrae/capstone-project/blob/master/modelling%202.ipynb#L262)

Before engineering new features, I ran a quick linear regression to see roughly what the cross-valdiated score was; at 0.22 I realised that this project needed a lot more work.

After engineering the all-important new features, my highest cross-validated score of 0.68, a huge increase.

By examining my coefficients I found that the variables: number of Spotify followers, artist familiarity and track longevity had the largest impact on track popularity.



## Evaluation

As I mentionned in the EDA section: you’re aiming to get your songs into automatically curated playlists or wonder how to get higher rankings in Spotify popularity index make sure you’re getting your listens now. That probably means it’s good to release songs frequently to stay relevant in the Spotify world.

The obvious trend is that the Billboard Hot 100 will continue to musically converge, a path that might just be the natural progression of popular culture. give it enough time and we’ll all be listening to the same thing.


## Plans for the Future
Due to time constraints I focused on the musical components and artist information derived from the Spotify API. However there are several other potentially brilliant predictors of popularity I can't wait to add:

#### LYRICS (GENIUS API)
#### NO. PRODUCERS (GENIUS API)
#### ARTIST LOCATION (WIKIPEDIA API)
#### INSTAGRAM / SOUNDCLOUD /FACEBOOK / TWITTER FOLLOWERS	

