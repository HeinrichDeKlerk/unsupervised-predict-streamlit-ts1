"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
import random
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',',delimiter=',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))

# def prediction_item(item_id):
#     """Map a given favourite movie to users within the
#        MovieLens dataset with the same preference.

#     Parameters
#     ----------
#     item_id : int
#         A MovieLens Movie ID.

#     Returns
#     -------
#     list
#         User IDs of users with similar high ratings for the given movie.

#     """
#     # Data preprosessing
#     reader = Reader(rating_scale=(0, 5))
#     load_df = Dataset.load_from_df(ratings_df,reader)
#     a_train = load_df.build_full_trainset()

#     predictions = []
#     for ui in a_train.all_users():
#         predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
#     return predictions

# def pred_movies(movie_list):
#     """Maps the given favourite movies selected within the app to corresponding
#     users within the MovieLens dataset.

#     Parameters
#     ----------
#     movie_list : list
#         Three favourite movies selected by the app user.

#     Returns
#     -------
#     list
#         User-ID's of users with similar high ratings for each movie.

#     """
#     # Store the id of users
#     id_store=[]
#     # For each movie selected by a user of the app,
#     # predict a corresponding user within the dataset with the highest rating
#     for i in movie_list:
#         predictions = prediction_item(item_id = i)
#         predictions.sort(key=lambda x: x.est, reverse=True)
#         # Take the top 10 user id's from each movie with highest rankings
#         for pred in predictions[:10]:
#             id_store.append(pred.uid)
#     # Return a list of user id's
#     return id_store
def proc_func():
    
    score = pd.merge(ratings[['userId','movieId','rating']], movies_df[['title',"movieId"]],on = "movieId")
    x = score.pivot_table(index=['title'], columns=['userId'], values='rating')  

    x_norm = x.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)

    x_norm.fillna(0, inplace=True)
    x_norm = x_norm.T
    x_norm = x_norm.loc[:, (x_norm != 0).any(axis=0)]
    x_sparse = sp.sparse.csr_matrix(x_norm.values)

    similarity = cosine_similarity(x_sparse.T) 
    df = pd.DataFrame(similarity, index = x_norm.columns,columns = x_norm.columns)
    
    return df

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.

def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    sims = proc_func()
    
    if movie_list[0] not in sims.columns:
        index_1 = pd.DataFrame()
    else:
        index_1 = pd.DataFrame(sims[movie_list[0]])
        index_1['similarity']= index_1[movie_list[0]]
        index_1 = pd.DataFrame(index_1, columns=['title','similarity'])
    
    if movie_list[1] not in sims.columns:
        index_2 = pd.DataFrame()
    else:
        index_2 = pd.DataFrame(sims[movie_list[1]]) 
        index_2['similarity'] = index_2[movie_list[1]]
        index_2 = pd.DataFrame(index_2, columns=['title','similarity'])
              
    if movie_list[2] not in sims.columns:
        index_3 = pd.DataFrame()
    else:
        index_3  = pd.DataFrame(sims[movie_list[2]])
        index_3['similarity'] = index_3[movie_list[2]]
        index_3 = pd.DataFrame(index_3, columns=['title','similarity'])

    similar_list = pd.concat([index_1, index_2, index_3])
              
    if similar_list.empty:
        #SOMETHING WENT WRONG
        print('Opps!! 404 Something Went Wrong')
    else:
        r_list = similar_list.sort_values('similarity', ascending=False)
        r_list = r_list[~(r_list['title'].isin(movie_list))]
        r_list = list(r_list[:top_n]['title'])
    return r_list
    