import numpy as np
from scipy.spatial.distance import cdist

__author__ = "threynaud"

# TODO: Handle the long tail phenomenon by replacing cdist.


class Recommender(object):
    """
    Recommender system object.
    Initialize it with a ratings matrix (with numerical values) and
    use the 'recommend' method to provide a recommendation for a given user.
    WARNING: This code in its current version does not adress the long tail
    phenomenon!
    """
    def __init__(self, R):
        """
        R: ratings matrix. list of lists or np.mat
        mean: mean rating for each user. NaNs are excluded. list
        R_centered: mean centered R. np.mat
        items_specified: list of items indices each user specified\
        a rating to. list of lists
        users_specified: list of users indices each item has a specified\
        rating for. list of lists
        users_similarity_mat:  S[i, j] is the similarity value\
        between users i and j. np.mat
        """
        self.R = R
        self.nb_users, self.nb_items = self.R.shape
        self.mean = self._compute_mean()
        self.R_centered = self._mean_center()
        self.items_specified = None
        self.users_specified = None
        self.items_similarity_mat = None
        self.users_similarity_mat = None

    def _clean_R(R):
        if type(R) == list:
            R = np.mat(R)
        elif type(R) == np.matrixlib.defmatrix.matrix:
            return R
        else:
            raise TypeError('R must be a list of lists or a np.mat!')

    def _compute_mean(self):
        mean = np.array([np.nanmean(self.R[u]) for u in range(self.nb_users)])
        return mean

    def _mean_center(self):
        R_centered = self.R - self.mean[:, np.newaxis]
        return R_centered

    def _compute_items_specified(self):
        specified = [[i[1] for i in np.argwhere(~np.isnan(self.R[u]))]
                     for u in range(self.nb_users)]
        self.items_specified = specified

    def _compute_users_specified(self):
        specified = [[u[0] for u in np.argwhere(~np.isnan(self.R[:, i]))]
                     for i in range(self.nb_items)]
        self.users_specified = specified

    def _compute_users_similarity_mat(self, metric, alpha, beta):
        S = np.array([[self.users_similarity(u, v,
                                             metric=metric,
                                             alpha=alpha,
                                             beta=beta)
                       for v in range(self.nb_users)]
                      for u in range(self.nb_users)])

        self.users_similarity_mat = S

    def _compute_items_similarity_mat(self, metric, alpha, beta):
        S = np.array([[self.items_similarity(i, j,
                                             metric=metric,
                                             alpha=alpha,
                                             beta=beta)
                       for j in range(self.nb_items)]
                      for i in range(self.nb_items)])

        self.items_similarity_mat = S

    def users_similarity(self, user1, user2,
                         metric='correlation', alpha=1, beta=1):
        """
            Returns the similarity value between two users. float.
            Inputs:
            user1, user2: indices of the two users. int.
            metric: Any metric supported by scipy. str or custom metric.
            alpha: Power of the similarity value returned. int or float.
            beta: Discount factor. int.
        """
        if self.items_specified is None:
            self._compute_items_specified()

        # Compute the items rated by both users.
        intersect = list(set(self.items_specified[user1]).intersection(
            set(self.items_specified[user2])))

        # If no common rating, similarity is null.
        if not intersect:
            return 0
        # Compute discount factor
        discount_factor = min(
            len(intersect), beta) / beta
        # Compute distance over the items in common.
        sim = float(1 - cdist(self.R_centered[user1, intersect],
                              self.R_centered[user2, intersect], metric))
        sim *= discount_factor
        return sim ** alpha

    def items_similarity(self, item1, item2,
                         metric='cosine', alpha=1, beta=1):
        """
            Returns the similarity value between two itmes. float.
            Inputs:
            item1, item2: indices of the two users. int.
            metric: Any metric supported by scipy. str or custom metric.
            alpha: Power of the similarity value returned. int or float.
            beta: Discount factor. int.
        """
        if self.users_specified is None:
            self._compute_users_specified()

        # Compute the users with ratings for both items.
        intersect = list(set(self.users_specified[item1]).intersection(
            set(self.users_specified[item2])))

        # If no common users, similarity is null.
        if not intersect:
            return 0
        # Compute discount factor
        discount_factor = min(
            len(intersect), beta) / beta
        # Compute distance over the users in common.
        sim = float(1 - cdist(np.squeeze(self.R_centered[intersect, item1]),
                              np.squeeze(self.R_centered[intersect, item2]),
                              metric))
        sim *= discount_factor
        return sim ** alpha

    def k_closest_users(self, user_tp, item_tp, k=2, threshold=0,
                        metric='correlation', alpha=1, beta=1):
        """
            Returns the top k users similar to user_tp who rated item_tp. list.
            Inputs:
            user_tp: target user index. int.
            item_tp: target item index. int.
            k: number of similar users to consider to predict the missing\
            rating. int.
            threshold: similarity value under which a user is\
            filtered out. float.
        """
        if self.users_similarity_mat is None:
            self._compute_users_similarity_mat(metric, alpha, beta)

        # List of users who rated item_tp
        users_rated_item = [i[0] for i in np.argwhere(
            ~np.isnan(self.R[:, item_tp]))]

        # Similarities between user_tp and every other users.
        similarity_of_users = self.users_similarity_mat[user_tp]

        # Rank users who rated item_tp from the most similar to the less.
        rank_similar_users = sorted(range(self.R.shape[0]),
                                    key=lambda a: similarity_of_users[a],
                                    reverse=True)

        rank_similar_users = [u for u in rank_similar_users
                              if u in users_rated_item
                              and similarity_of_users[u] > threshold]

        # Returns list of the top k similar users who rated item_tp.
        return rank_similar_users[:k]

    def k_closest_items(self, user_tp, item_tp, k=2, threshold=0,
                        metric='cosine', alpha=1, beta=1):
        """
            Returns the top k users similar to user_tp who rated item_tp. list.
            Inputs:
            user_tp: target user index. int.
            item_tp: target item index. int.
            k: number of similar items to consider to predict the missing\
            rating. int.
            threshold: similarity value under which a user is\
            filtered out. float.
        """
        if self.items_similarity_mat is None:
            self._compute_items_similarity_mat(metric, alpha, beta)

        # List of items rated by user_tp
        items_rated_user = [i[1] for i in np.argwhere(~np.isnan(self.R[user_tp]))]

        # Similarities between item_tp and every other items.
        similarity_of_items = self.items_similarity_mat[item_tp]

        # Rank items rated by user_tp from the most similar to the less.
        rank_similar_items = sorted(range(self.nb_items),
                                    key=lambda a: similarity_of_items[a],
                                    reverse=True)

        rank_similar_items = [i for i in rank_similar_items
                              if i in items_rated_user
                              and similarity_of_items[i] > threshold]

        # Returns list of the top k similar items rated by user_tp.
        return rank_similar_items[:k]

    def predict_rating(self, user_tp, item_tp, k=2, algo='user',
                       threshold=0, metric='correlation', alpha=1, beta=1):
        """
            Returns the predicted rating for user_tp and item_tp. float
            Inputs:
            user_tp: target user index. int.
            item_tp: target item index. int.
            k: number of similar users to consider to predict the missing\
            rating. int.
        """

        if algo == 'user':
            close_users = self.k_closest_users(user_tp, item_tp, k,
                                               threshold=threshold,
                                               metric=metric,
                                               alpha=alpha, beta=beta)
            r_pred = (self.mean[user_tp] +
                      ((sum(
                            [self.users_similarity_mat[user_tp][v] *
                             self.R_centered[v, item_tp]
                             for v in close_users]) /
                        sum([abs(self.users_similarity_mat[user_tp][v])
                            for v in close_users]))))
            return r_pred

        elif algo == 'item':
            close_items = self.k_closest_items(user_tp, item_tp, k,
                                               threshold=threshold,
                                               metric=metric,
                                               alpha=alpha, beta=beta)
            
            r_pred = ((sum([self.items_similarity_mat[j][item_tp] *
                            self.R[user_tp, j]
                            for j in close_items]) /
                       sum([abs(self.items_similarity_mat[j][item_tp])
                            for j in close_items])))
            return r_pred

        else:
            raise ValueError("'algo' can only take 'user' or 'item' as values!")

    def recommend(self, user, k=2, algo='user',
                  nb_to_recommend=1, threshold=0,
                  metric='correlation', alpha=1, beta=1):
        """
            Returns the top recommendations for a given user.\
            List of tuples (item, predicted rating).
            Inputs:
            user: user index we want to make recommendation to.
            k: number of similar users to consider to predict the missing\
                rating. int.
            nb_to_recommend: number of top recommendations to return. int.
        """

        # List of items where rating is missing.
        items_tp = [i[1] for i in np.argwhere(np.isnan(self.R[user]))]
        # Predict rating for each missing value.
        predicted_ratings = [(item,
                              self.predict_rating(user, item, k,
                                                  algo=algo,
                                                  threshold=threshold,
                                                  metric=metric,
                                                  alpha=alpha, beta=beta))
                             for item in items_tp]
        # Rank them.
        top_k_items = sorted(predicted_ratings, key=lambda k: k[1],
                             reverse=True)[:nb_to_recommend]
        return top_k_items
