import numpy as np
from scipy.spatial.distance import cdist


def users_similarity(R, specified, user1, user2, metric='correlation',
                     power=1, discount_threshold=1):
    """ Returns the similarity value between two users. float.
        Inputs:
        R: matrix of ratings. 2-dimension iterable.
        specified: item indices for which ratings have been specified by each user. 2-dimension iterable.
        user1, user2: indexes of the two user we want to compute the similarity between. int.
        metric: Any metric supported by scipy. str or custom metric.
        power: change value to exponentiate the value of the similarity returned. int or float.
    """

    # Compute the items rated by both users.
    intersect = list(set(specified[user1]).intersection(set(specified[user2])))

    # If no common rating, similarity is null.
    if not intersect:
        return 0
    # Compute discount factor
    discount_factor = min(
        len(intersect), discount_threshold) / discount_threshold
    # Compute distance over the items in common.
    sim = float(1 - cdist(R[user1, intersect], R[user2, intersect], metric))
    sim *= discount_factor
    return sim ** power


def k_closest_users(R, S, user_tp, item_tp, k, threshold=0):
    """ Returns the top k users similar to a user_tp who rated item_tp as well.
        Inputs:
        R: matrix of ratings. 2-dimension iterable.
        S: similarity matrix. 2-dimension iterable.
        user_tp: given user for which we want to predict missing ratings. int.
        item_tp: given item we want to predict the rating for. int.
        k: number of similar users to consider in order to predict the missing rating. int.
        threshold: value of similarity under which a user is filtered out. float.
    """

    # List of users who rated item_tp
    users_rated_item = [i[0] for i in np.argwhere(~np.isnan(R[:, item_tp]))]
    # Similarities between user_tp and every other users.
    similarity_of_users = S[user_tp]
    # Rank users who rated item_tp from the most similar to the less similar.
    rank_similar_users = sorted(
        range(R.shape[0]), key=lambda a: similarity_of_users[a], reverse=True)
    rank_similar_users = [
        u for u in rank_similar_users if u in users_rated_item and similarity_of_users[u] > threshold]
    # Returns list of the top k similar users who rated item_tp.
    return rank_similar_users[: k]


def predict_rating(R, S, R_centered, mean, user_tp, item_tp, k):
    """ Returns the predicted rating for user_tp and item_tp. float
        Inputs:
        R: matrix of ratings. 2-dimension iterable.
        S: similarity matrix. 2-dimension iterable.
        R_centered: mean cetered ratings matrix. 2-dimension iterable.
        user_tp: given user for which we want to predict missing ratings. int.
        item_tp: given item we want to predict the rating for. int.
        k: number of similar users to consider in order to predict the missing rating. int.
    """

    close_users = k_closest_users(R, S, user_tp, item_tp, k)
    r_pred = mean[user_tp] + (
        (sum([S[user_tp][v] * R_centered[v, item_tp] for v in close_users]) / sum([abs(S[user_tp][v]) for v in close_users])))
    return r_pred


def recommend(R, S, user, k, nb_to_recommend):
    """ Returns the top recommendations for a given user. List of tuples (item, predicted rating).
        R: matrix of ratings. 2-dimension iterable.
        S: similarity matrix. 2-dimension iterable.
        user: user we want to make recommendation to.
        k: number of similar users to consider in order to predict the missing rating. int.
        nb_to_recommend: number of top recommendations to return. int.
    """

    # List of items where rating is missing.
    items_tp = [i[1] for i in np.argwhere(np.isnan(R[2]))]
    # Predict rating for each missing value.
    predicted_ratings = [(item, predict_rating(
        R, S, R_centered, mean, user, item, k)) for item in items_tp]
    # Rank them.
    top_k_items = sorted(predicted_ratings, key=lambda k: k[
                         1], reverse=True)[:nb_to_recommend]
    return top_k_items


# ------ MAKING THE RECOMMENDATIONS -----
# The rating matrix R.
# We use nan to specify a missing value.
R = np.mat([[7, 6, 7, 4, 5, 4],
            [6, 7, np.nan, 4, 3, 4],
            [np.nan, 3, 3, 1, 1, np.nan],
            [1, 2, 2, 3, 3, 4],
            [1, np.nan, 1, 2, 3, 3]])

USER_TO_RECOMMEND = 2
NB_RECOMMENDATIONS = 2

nb_users, nb_items = R.shape

# mean is an array containing the mean rating for each user. NaNs are not
# taken into consideration.
mean = np.array([np.nanmean(R[u]) for u in range(nb_users)])

# specified is a list of lists. It contains the set of item indices for which
# ratings have been specified by each user.
specified = [[i[1]
              for i in np.argwhere(~np.isnan(R[u]))] for u in range(nb_users)]

# S is the similarity matrix. S[i, j] is the similarity value between users
# i and j.
S = np.array([[users_similarity(R, specified, u, v)
               for v in range(nb_users)] for u in range(nb_users)])

# R_centered contains the mean centered ratings for each user.
R_centered = R - mean[:, np.newaxis]

# Extract recommended items and their corresponding predicted ratings in
# two different lists.
items, ratings = zip(
    *recommend(R, S, USER_TO_RECOMMEND, 2, NB_RECOMMENDATIONS))

# Et voilà!
print("The items recommended for user {0} are : {1}".format(
    USER_TO_RECOMMEND, items))
print("The corresponding predicted ratings are : {0}".format(
    ["{0:.2f}".format(round(i, 2)) for i in ratings]))
