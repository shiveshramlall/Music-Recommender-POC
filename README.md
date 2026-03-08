# Music-Recommender-POC
A Music Recommender that uses Collaborative Filtering to provide recommendations and a Content Based Filter to help sort them by similarity to the compared playlist.

## Purpose of Repo
Create a basic Streamlit application to demo the functionality of the recommenders

## Tech Overview
- Tensorflow Recommenders [1]
- scikit-learn [2]
- Mlxtend [3]
- Spotipy Web API [4]
- Streamlit [5]

## Project Intro

I wanted to utilize collaborative filtering [1] to provide song recommendations. 

One needs typically user ratings to achieve this, and this data isn't easily available. As a workaround, I used the "Million Playlist" [6] dataset. 

Using this data I have an implicit rating sytem, in the sense that the playlist will act like a user, and the songs in it will be of interest. 

## Updates

- The Spotipy API call to collect audio features has been deprecated [7]

- Therefore, content based filtering is no longer possible unfortunately (for gathering most similar song recommendations relative to the playlist)

## Notebooks

### association_analysis.ipynb
- Some EDA work to get familiar with the data and the spotify API to find any interesting relationships in terms of frequent item sets [3].

### playlist_recommendor_create_model.ipynb
- Creates the Collaborative Filtering model by taking a subset of the "Million Playlist" [1] data (for memory reasons). 
- Uses the Spotipy library [4] to also grab extra data that will be more applicable to the user.

### playlist_recommendor_load_model.ipynb
- Loads the model and its recommendations
- Makes use of the py files in the "modules" folder to clean outputs
- Additionally it also has steps to use a content based filter [2] in order to help sort recommendations from the model in terms of similarity to the playlist

## Python Files

### content_recommender.py
- Uses sklearn [2] to build a recommendor based using a content based approach. 
> Deprecated due to [7] - retrieval of audio features are no longer possible

### recommender_methods.py
- Methods to cleanse the outputs of the Collaborative Filtering recommendations.

### spotify_methods.py
- Methods to utilize the spotipy library [4].
- Requires a spotify_auth.py file that sets the environment variables used here for autentication.
> Due to [7], retrieval of audio features is no longer possible

## Streamlit App
- A simple interface to provide reccomendations to users that we have manually added from the Spotipy API [4]. 
- Provides a shuffled list from the Collaborative Filtering model, or allows you to rank them by similarity
- Getting the similar rankings will take a while due to having to get song metadata from the Spotipy API [4]. The Million Playlist [6] data does not have the features of songs.
> Due to [7], retrieval of audio features is no longer possible

### Example Recommendations (shuffled)
![Image not found](<resources/Screenshot - Recommendations.png>)
- The above shows recommendations for a nicher use case, acoustic covers or songs. Anything that can be considered relaxed and instrumental in nature

### Example Recommendations (similar)
![Image not found](<resources/Screenshot - Similar Filter.png>)
- The above ranks the recommendations by similarity to the overall tone of the original playlist
- In the example, we see a playlist for 2000s hits, with similar throwbacks being recommended to fit the theme. 
- The original playlist has songs closer to the end of 2010 as well, which is why we can see songs from 2010 or 2011 being associated with it

## Caching Song Features
- The Content Based Recommender saves the data on songs to a file, "song_features.csv" to make it slightly faster in future runs. 
- A future optimization update would be to move away from a csv file to something a lot faster. The bottleneck will remain to be the downloading of song features from Spotipy. 

# References
[1] - "Tensorflow Recommenders" - https://www.tensorflow.org/recommenders/examples/quickstart

[2] - "scikit-learn" - https://scikit-learn.org/stable/

[3] - "Mlxtend" - https://rasbt.github.io/mlxtend/

[4] - "Spotipy" - https://github.com/spotipy-dev/spotipy

[5] - "Streamlit" - https://streamlit.io/

[6] - "Million Playlist" - https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files

[7] - "Spotipy Audio Features" - https://developer.spotify.com/documentation/web-api/reference/get-audio-features