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

# Visual dependancies
from PIL import Image
# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview","Exploratory Data Analysis","How a Recommender System Works"]

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


    # -------------- EDA PAGE -------------------------------------------
    if page_selection == "Exploratory Data Analysis":
        st.title("Exploratory Data Aanalysis")
        st.info("On this page we will Explore the data and relay any insights we have gained from it")

    # -------------- HOW IT WORKS PAGE ----------------------------------
    if page_selection == "How a Recommender System Works":
        
        st.title("How a Recommender System Works")
        rec_image = Image.open("resources/imgs/rec_eng_img.jpg.jpeg")
        st.image(rec_image, use_column_width=True)
        
        st.info("Here you wil find some simple explanations on how a recommender system works.")
        st.markdown("A Recommender System/Engine can suggest items or actions of interest, or in our case, movie recommendations to a user, based on their similarity to other users.<br> By similarity one means how similar one user is to another, based on their likes and dislikes, their demographic information, their preferred genre's, or the rating that they give items.", unsafe_allow_html=True)
        st.markdown("We chose to mainly focuss on a **Collaborative-Based** Recommender.<br> The recommender system we created is one that will provide movie recommendations to a user(user1), by having them choose 3 movies that they like from a list, and from that choice we calculate their similarity to other users(user2,5 and 6) who also rated those movie's highly.<br> We then see which other movies Users 2, 5 and 6 have rated highly that User 1 has not seen yet, and recommend those to User 1.<br> <img src='https://miro.medium.com/max/2728/1*x8gTiprhLs7zflmEn1UjAQ.png' alt='colab' width='550' height='450'/>", unsafe_allow_html=True)
        
        st.markdown("There is also a **Content-Based** Recommender system, which instead of the user ratings, takes into account the content of the films, and how similar that content is to the content of other films, such as: Genre, duration, actors, release year, director, demographics and more.<br> <img src='https://miro.medium.com/max/1642/1*BME1JjIlBEAI9BV5pOO5Mg.png' alt='content' width='400' height='500'/>", unsafe_allow_html=True)

        st.markdown("The drawback to this method is that it does not always take into account the _'Humanity'_ aspect, where users are likely to belong to more than one 'demographic' into which a Content-Based System creates it's similarities.")
if __name__ == '__main__':
    main()
