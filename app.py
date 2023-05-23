import streamlit as st
import os
import pickle as pk
import json
import random
import numpy as np
import pandas as pd

from modules import content_recommender as cr
from modules import spotify_methods as sm
from modules import recommender_methods as rm

loaded = rm.load_model()

@st.experimental_memo
def get_users():

    with open('model\\playlists_users_ref.json') as json_file:
        users_dict = json.load(json_file)

    return users_dict

@st.experimental_memo
def get_pids():
    with open('model\\playlists_id_ref.json') as json_file:
        pids_dict = json.load(json_file)

    return pids_dict

@st.experimental_memo
def get_training_data():
    with open('model\\extra_playlist_songs.json') as json_file:
        spotify_playlists = json.load(json_file)

    return spotify_playlists

def print_playlist(playlist, limit=None):
    if not limit:
        limit = len(playlist)
    for song in playlist[0: limit]:
        st.markdown(song)


training_data = get_training_data()
users_dict = get_users()
pids_dict = get_pids()

st.title("Music Recommender POC")

users_dict.pop("anonymous")
users_ids = list(users_dict.keys())
sel_user = st.selectbox("Select a User", users_ids)

pid_names = []
for pid, name in users_dict.get(sel_user).items():
    pid_names.append(pid+" -- "+name)

sel_playlist = st.selectbox("Select a Playlist", pid_names, index=0)
sel_playlist = sel_playlist[0: sel_playlist.find("--")-1]
orig_playlist = rm.show_playlist(sel_playlist, training_data)

with st.expander("Playlist: "+pids_dict.get(sel_playlist).get("name")):
    preview = st.checkbox('Limit Preview')
    limit = 15 if preview else len(orig_playlist)
    print_playlist(orig_playlist, limit)

col1, col2 = st.columns(2)
with col1:
    shuffle_search = st.button("Shuffle Recommendations --- (Collaborative Filtering)")
with col2:
    similar_search = st.button("Similar Recommendations --- (Collaborative Filtering + Content Based - Takes longer)")

if(shuffle_search):

    st.markdown("## Results From Shuffle Search")

    scores, all_rec_songs = loaded([str(sel_playlist)])
    selected_recommendations = rm.decode_and_select(all_rec_songs, 1000)
    recommendations = rm.remove_known_positives(selected_recommendations, orig_playlist)
    np.random.shuffle(recommendations)

if(similar_search):

    with st.spinner('Getting and downloading song meta data... please wait'):
        st.markdown("## Results From Similar Search")
        
        scores, all_rec_songs = loaded([str(sel_playlist)])
        selected_recommendations = rm.decode_and_select(all_rec_songs, 1000)
        de_duped_rec = rm.remove_known_positives(selected_recommendations, orig_playlist)

        cached_songs = pd.read_csv('./song_features.csv')
        to_cache_items=[]

        sr_features = []
        for sr in de_duped_rec:

            search_result, to_cache = sm.get_song_data(sr, cached_songs)
            if(search_result is not None):
                sr_features += [search_result]
            if(search_result is None):
                print(f'''Not found: {sr}''')
            if(to_cache):
                to_cache_items += [search_result]

        op_features = []
        for op in orig_playlist:
            
            search_result, to_cache = sm.get_song_data(op, cached_songs)
            if(search_result is not None):
                op_features += [search_result]
            if(search_result is None):
                print(f'''Not found: {op}''')
            if(to_cache):
                to_cache_items += [search_result]
        
        new_cached_songs = pd.DataFrame(to_cache_items)
        combine_frames = [cached_songs, new_cached_songs]

        all_cached_songs = pd.concat(combine_frames)
        all_cached_songs.to_csv(path_or_buf="song_features.csv", index=False)
        song_cluster_pipeline, number_cols = cr.cluster_pipeline(sr_features, 20)
        recommendations = cr.recommend_songs(op_features, sr_features, song_cluster_pipeline, number_cols)

if(similar_search or shuffle_search):

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("# Original Playlist")
        ([st.markdown(o) for o in orig_playlist])

    with col4:
        st.markdown("# Recommendations")
        ([st.markdown(r) for r in recommendations])






