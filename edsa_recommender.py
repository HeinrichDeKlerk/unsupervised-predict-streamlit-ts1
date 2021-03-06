
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
    st.sidebar.title('Movie Recommender App')
    st.sidebar.markdown('''
        ## **Pages**\n
        * Recommender System\n
        * Data Description\n
        * Exploratory Data Analysis\n
        * How a Recommender System Works\n
        ## Choose a page in the selectbox below:
        ''')
    page_options = ["Recommender System","Data Description","Exploratory Data Analysis","How a Recommender System Works"]


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

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------------

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
        st.markdown('### Distribution of Ratings')
        rate_dist = movie_rate_df.groupby('rating').size()
        rate_dist_df = rate_dist.reset_index()
        fig = px.bar(rate_dist_df, y=0, x='rating',
                    labels={'rating':"Rating", '0':'Count'},
                    color=0)
        st.plotly_chart(fig)

        st.markdown('Looking at the rating distribution we can see that most Users are generous when rating a Film, with the majority of ratings 3 or more stars')
        

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
        st.markdown('Most plots here do not show anything of value except the final bottom right one.<br> This shows a very clear correlation between the movie length and ratings.',unsafe_allow_html=True)
        st.markdown('Movies that are longer than average seem to get lower reviews than shorter movies, which can give some insight into how most people view movie length.')
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

        st.markdown("## Our approach")
        st.markdown("""
            >We chose _Singular Value Decomposition_ (SVD) as our base model (this algorithm was made famous when it was used to win a Netflix Recommender challenge with Collaborative Filtering)\n
>SVD turns our very sparse matrix into a low ranking matrix (reducing the dimensionality) with userID and ItemID or simply known as factors.\n
>We then use this matrix and the SVD Algorithm to predict any missing parts of the matrix, so that each user has a corresponding rating for an Item.\n
>After this, to actually make recommendations to a user we ask them to choose 3 of their favorite movies from a list. Usingtheir choices we then calculate how similar they are to other users who also rated those movies highly. (To accomplish this we used Cosine Similarity to create a similarity matrix and calculated their similar users from this)\n
>We then take the top rated movies from each of the similar users and calculate which top 10 we want to recommend, and return that as our recommendation to the user.

            """)
        st.markdown("## Final thoughts")
        st.markdown("""
        >Why not content based filtering?\n
>Besides the fact that content based filtering is more resource heavy and having large amounts of data to process therefore taking far longer.\n
>Humans in general would prefer to watch a movie someone else has seen and with a similar taste to theirs (this being collaborative).\n
>Consider this, we might all like action movies , but not all action movies are great!\n
>This is where content based filtering fails, It might recommend a movie because you last watched something similar but only to find you hate that movie or worse you might like both horror and comedy , but content based filtering will most likely only recommend one genre over the other.\n
>This means that our chosen algorithm might need a larger dataset, however it is more varied, and willl most likely give a range of recommendation.\n
>As powerful as machine learning is , We learned that in our task of making prediction, what we were actaully doing was predictiong emotions based on past behaviuor of what a user would rate a certain movie.\n
>That is a very complex task and will never be 100% correct all the time. Humans are complex, and their tastes vary widely, not always conforming into one _Category_.\n
>This recommender system we built is not only useful in just predictiong movie ratings but a recommender system can be usedin a number of other tasks aswell, like music or book recommendations, Online purchasing websites recommending products for sale or any number of other methods.
>        """)

    st.sidebar.title('About')
    st.sidebar.info(
            """
            This App is maintained by EDSA students. It serves as a project for
            an unsupervised learning sprint, by deploying a recommender engine.

            **Authors:**\n
            Heinrich de Klerk\n
            Michael Ilic\n
            Rolivhuwa Malise\n
            Rirhandzu Mahlaule\n
            Nceba Mandlana\n
            Siyabonga Mtshemla\n

    """
        )

if __name__ == '__main__':
    main()
