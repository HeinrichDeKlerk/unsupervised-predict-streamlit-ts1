"""
    Streamlit webserver-based Recommender Engine.
    Author: Explore Data Science Academy.
    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.
    NB: !! Do not remove/modify the code delimited by dashes !!
    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------
    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.
	For further help with the Streamlit framework, see:
	https://docs.streamlit.io/en/latest/
"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
from markdown import markdown
from pathlib import Path

# Visual dependancies
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
s3_path = Path('''../unsupervised_data/unsupervised_movie_data/''')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Data Description","Solution Overview","Exploratory Data Analysis","How a Recommender System Works"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


    # -------------- Data Description Page ------------------------------
    if page_selection == "Data Description":
        st.title("Data Description")
        st.subheader("This recommender makes use of data from the MovieLens recommendation service")

        data_descrip = markdown(open('resources/md_files/movielens_data_descrip.md').read())
        st.markdown(data_descrip, unsafe_allow_html=True)

    # -------------- EDA PAGE -------------------------------------------
    if page_selection == "Exploratory Data Analysis":
        st.title("Exploratory Data Aanalysis")
        st.info("On this page we will Explore the data and relay any insights we have gained from it")

        data_descrip = markdown(open('resources/md_files/movielens_data_descrip.md').read())
        st.markdown(data_descrip, unsafe_allow_html=True)
        
        m_df = pd.read_csv(s3_path/'movies.csv')
        r_df = pd.read_csv(s3_path/'train.csv')

        st.write(m_df.head())
        st.write(r_df.head())
        movie_rate_df = pd.merge(r_df, m_df, on="movieId")

        # correlation matrix
        fig = plt.figure(figsize = (15, 10))
        ax = fig.add_subplot()
        ax.imshow(r_df.corr(), cmap = 'viridis', interpolation='nearest')
        ax.set_title("Correlation between features")
        st.pyplot()

        # Rating distribution
        fig, ax = plt.subplots(figsize=(10,5))
        graph = sns.countplot(x='rating', data=movie_rate_df, ax=ax)
        plt.title('Rating distribution')
        plt.xlabel("Rating")
        plt.ylabel("Count of Ratings")
        st.pyplot()

        st.write(np.mean(r_df['rating']))


    # -------------- HOW IT WORKS PAGE ----------------------------------
    if page_selection == "How a Recommender System Works":
        
        rec_image = Image.open("resources/imgs/rec_eng_img.jpg.jpeg")
        st.image(rec_image, use_column_width=True)
        
        st.title("How a Recommender System Works")
        st.info("Here you wil find some simple explanations on how a recommender system works.")



if __name__ == '__main__':
    main()
