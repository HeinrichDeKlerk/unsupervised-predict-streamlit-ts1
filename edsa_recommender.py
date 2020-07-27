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

# Visual dependancies
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
        
        m_df = pd.read_csv(s3_path/'movies.csv')
        r_df = pd.read_csv(s3_path/'train.csv')

        st.markdown('### Movie information Data') 
        st.write(m_df.head())
        st.write('The Movie Data has {} rows and {} columns'.format(len(m_df.axes[0]), len(m_df.axes[1])))

        st.markdown('### Rating Data')
        st.write(r_df.head())
        st.write('The Rating Data has {} rows and {} columns'.format(len(r_df.axes[0]), len(r_df.axes[1])))

        movie_rate_df = pd.merge(r_df, m_df, on="movieId")

        st.markdown('### Correlation between Rating Data')
        corr = r_df.corr()

        # create a mask and only show half of the cells 
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        fig, ax = plt.subplots(figsize=(10,5))
        # plot the data using seaborn
        graph = sns.heatmap(corr, 
                         mask = mask, 
                         vmax = 0.3, 
                         #square = True,  
                         cmap = "viridis")

        plt.title("Correlation between features")
        st.pyplot()

        st.markdown('We can see a strong correlation between _timestamp_ and _movieId_.<br> It appears that movies with the lowest ratings last for around 1.5 hours, which implies that the rating users give a film can be dependant on the length of the film.', unsafe_allow_html=True)

        # Rating distribution
        def rat_distplot(df):
            fig, ax = plt.subplots(figsize=(10,5))
            graph = sns.countplot(x='rating', data=df, ax=ax)
            plt.title('Rating distribution')
            plt.xlabel("Rating")
            plt.ylabel("Count of Ratings")
            return

        st.markdown('### Distribution of Ratings')
        rat_distplot(movie_rate_df)
        st.pyplot()

        st.markdown('Looking at the rating distribution we can see that most Users are generous when rating a Film, with the majority of ratings 3 or more stars')
        
        most_rate = movie_rate_df.groupby('title').size().sort_values(ascending=False)[:10]
        most_rate_df = most_rate.reset_index()
        fig = px.bar(most_rate_df, y='title', x=0,
                    labels={'title':"Movie Title", 0:'Count'},
                    color=0)
        st.plotly_chart(fig)


        st.markdown("### Pairwise plot of Rating Data")
        pairplot = Image.open('resources/imgs/pairplot.jpg')
        st.image(pairplot, use_column_width=True)
    # -------------- HOW IT WORKS PAGE ----------------------------------
    if page_selection == "How a Recommender System Works":
        
        rec_image = Image.open("resources/imgs/rec_eng_img.jpg.jpeg")
        st.image(rec_image, use_column_width=True)
        
        st.title("How a Recommender System Works")
        st.info("Here you wil find some simple explanations on how a recommender system works.")

        st.markdown("## What is a Recommender System?")
        st.markdown(">Put simply, a recommender system is used to allow a service provider to build a catalogue of items or suggestions that they want to present to a user.<br> This allows them to offer relevant service to their users without overloading them with information that they may not want to see or sift through themselves.<br> In this era of technology and bountifull information it is very important that a user is given relevant information, but also in manageable amounts, about content, as there is too much for a user to give attention to individually.", unsafe_allow_html=True)
        st.markdown("A Recommender System/Engine can suggest items or actions of interest, or in our case, movie recommendations to a user, based on their similarity to other users.<br> By similarity one means how similar one user is to another, based on their likes and dislikes, their demographic information, their preferred genre's, or the rating that they give items.", unsafe_allow_html=True)

        if st.button('Recommender Types'):
            st.markdown("We chose to mainly focuss on a **Collaborative-Based** Recommender.<br> The recommender system we created is one that will provide movie recommendations to a user(user1), by having them choose 3 movies that they like from a list, and from that choice we calculate their similarity to other users(user2,5 and 6) who also rated those movie's highly.<br> We then see which other movies Users 2, 5 and 6 have rated highly that User 1 has not seen yet, and recommend those to User 1.<br> <img src='https://miro.medium.com/max/2728/1*x8gTiprhLs7zflmEn1UjAQ.png' alt='colab' width='550' height='450'/>", unsafe_allow_html=True)
            
            st.markdown("There is also a **Content-Based** Recommender system, which instead of the user ratings, takes into account the content of the films, and how similar that content is to the content of other films, such as: Genre, duration, actors, release year, director, demographics and more.<br> <img src='https://miro.medium.com/max/1642/1*BME1JjIlBEAI9BV5pOO5Mg.png' alt='content' width='400' height='500'/>", unsafe_allow_html=True)

            st.markdown("The drawback to this method is that it does not always take into account the _'Humanity'_ aspect, where users are likely to belong to more than one 'demographic' into which a Content-Based System creates it's similarities.")

if __name__ == '__main__':
    main()
