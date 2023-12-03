import numpy as np
import pandas as pd


def recommend_svd(user_id, df_ratings, movies_ids, svd, num_of_recs=5):
    recs = {}

    # filter out already rated movies
    movies_rated_ids = df_ratings.query(f'user_id == {user_id}')['movie_id'].tolist()
    movies_to_predict_ids = [id for id in movies_ids if id not in movies_rated_ids]
    # predict ratings for the rest movies
    for movie_id in movies_to_predict_ids:
        prediction = svd.predict(user_id, movie_id)
        recs[movie_id] = prediction.est

    recs_sorted = dict(sorted(recs.items(), key=lambda item: item[1], reverse = True))

    recs = list(recs_sorted.items())[:num_of_recs]

    return recs, recs_sorted


def recommend_hybrid(recs, user_id, df_ratings, df_users_similarity, similarity_threshold, num_of_recs=5):
    users_similarities = pd.eval(df_users_similarity.loc[user_id, 'similar_ids'])
    # take the similar users with cosine similarity >= similarity_threshold
    similar_users_ids = [tuple[0] for tuple in users_similarities if tuple[1] > similarity_threshold]
    similar_users_ratings = df_ratings[df_ratings['user_id'].isin(similar_users_ids)]

    for movie_id, rating_svd in recs.items():
        # get relevant ratings, i.e., the specified movie rating from users in similar_users_ids
        relevant_ratings = similar_users_ratings.query(f'movie_id == {movie_id}')
        relevant_ratings = relevant_ratings['rating'].tolist()
        num_relevant_ratings = np.count_nonzero(relevant_ratings)

        # Calculate the weighted rating
        if relevant_ratings:
            weighted_rating = (rating_svd + sum(relevant_ratings)) / (num_relevant_ratings + 1)
        else:
            weighted_rating = rating_svd
        
        recs[movie_id] = weighted_rating

    recs_sorted = dict(sorted(recs.items(), key=lambda item: item[1], reverse = True))

    recs = list(recs_sorted.items())[:num_of_recs]

    return recs, recs_sorted


def print_recs(df_items, recs):
    print(f'Recommendations for the user:')
    for i, rec in enumerate(recs):
        movie_id, predicted_rating = rec
        movie_title = df_items.query(f'movie_id == {movie_id}')['title'].iloc[0]
        print(f"{i+1}. {movie_title}.")
        print(f"Predicted rating: {predicted_rating}\n")