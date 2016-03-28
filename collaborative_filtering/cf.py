import numpy as np
import time
import operator
from collections import defaultdict
import argparse
import csv

def parse_arguments():
    """Code for parsing command line arguments
    """
    parser = argparse.ArgumentParser(description = 'Parsing filenames.')
    parser.add_argument('--train', nargs = 1, required = True)
    parser.add_argument('--test', nargs = 1, required = True)
    args = vars(parser.parse_args())
    return args
    
def read_file(filename):
    """Read in file with format: MovieID, CustomerID, Rating
    :param filename: name of txt file
    :return: two dicts with keys = (CustomerID, MovieID), (MovieID, CustomerID) and value = rating
    """
    user_dict = defaultdict(dict)
    movie_dict = defaultdict(dict)
    with open(filename, 'r') as file:
        for line in file:
            split_line = line.split(',')
            (movie, user, rating) = split_line
            user_dict[user][movie] = float(rating)
            movie_dict[movie][user] = float(rating)    
    return(user_dict, movie_dict)

def read_testing_file(filename):
    """Read in test file with format: MovieID, CustomerID, Rating
    :param filename: name of test file
    :return: dict with key = (movie, user) and value = rating
    """
    test_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            split_line = line.split(',')
            (movie, user, rating) = split_line
            test_dict[(movie, user)] = rating
    return(test_dict)

def calculate_similarity(user_dict, movie_dict, i, j):
    """Calculates pearson coefficient between two users 
    :param user_dict: dictionary where key = (CustomerID, MovieID) and value = rating
    :param movie_dict: dictionary where key = (MovieID, CustomerID) and value = rating
    :param i,j: user ids
    :return: pearson coefficient
    """
    ratings_i = []
    ratings_j = []
    w = 0
    movies_i = set(user_dict.get(i).keys())
    movies_j = set(user_dict.get(j).keys())
    movies_intersection = movies_i.intersection(movies_j)
    if len(movies_intersection) == 0 or len(movies_intersection) == 1:
        return w
    for key in movies_intersection:
        if user_dict[j][key]:
            ratings_i.append(user_dict[i][key])
            ratings_j.append(user_dict[j][key])
    mean_i = np.mean(user_dict.get(i).values())
    mean_j = np.mean(user_dict.get(j).values())
    numerator = sum((ratings_i - mean_i)*(ratings_j - mean_j))
    denominator = np.sqrt(sum((ratings_i-mean_i)**2)*sum((ratings_j-mean_j)**2))
    if denominator != 0:
        w = float(numerator)/denominator
    return w

def predict_unknown_user(user_dict, movie_dict, k):
    """Return mean of all ratings for the known movie as prediction for user
    :param user_dict: dict where key = (CustomerID, MovieID) and value = rating
    :param movie_dict: dict where key = (MovieID, CustomerID) and value = rating
    :return: prediction for an unknown user, known movie
    """
    for k in movie_dict.keys():
        prediction = round(np.mean(movie_dict.get(k).values()), 2)
    return prediction
    
def predict_unknown_movie(user_dict, movie_dict, i):
    """Return mean of all ratings for the known user as prediction for movie
    :param user_dict: dict where key = (CustomerID, MovieID) and value = rating
    :param movie_dict: dict where key = (MovieID, CustomerID) and value = rating
    :return: prediction for an unknown movie, known user
    """
    for i in user_dict.keys():
        prediction = np.mean(user_dict.get(i).values())
    return prediction

def predict_both_unknown(user_dict):
    """Return mean of all ratings for an unknown user and movie
    :param user_dict: dict where key = (CustomerID, MovieID) and value = rating
    :return: prediction for unknown movie, unknown user
    """
    ratings = []
    for user in user_dict.keys():
        for rating in user_dict.get(user):
            ratings.append(rating)
    prediction = np.mean(ratings)
    return prediction

def predict_both_known(user_dict, movie_dict, i, k):
    """Predicts rating for a known user and movie using similarity 
    :param user_dict: dict where key = (CustomerID, MovieID) and value = rating
    :param movie_dict: dict where key = (MovieID, CustomerID) and value = rating
    :param i: user id
    :param k: movie id
    :return: prediction for known user and movie
    """
    u_k = set(movie_dict.get(k).keys())
    weights = []
    differences = []
    means = {}
    for key in user_dict.keys():
        means[key] = np.mean(user_dict.get(key).values())
    mean_i = means.get(i)
    for user in u_k:
        w = calculate_similarity(user_dict, movie_dict, i, user)      
        weights.append(w)
        mean_j = means.get(user)
        rating_j = user_dict[user][k]
        differences.append(w*(rating_j-mean_j))
    if (sum(np.absolute(weights))) != 0:
        multiplier = 1.0/(sum(np.absolute(weights)))
    else:
        multiplier = 0.0
    prediction = mean_i + (multiplier*sum(differences))
    return prediction

def predict_rating(user_dict, movie_dict, i, k):
    """Predicts rating for a given user and movie (regardless if user/movie is known)
    :param user_dict: dict where key = (CustomerID, MovieID) and value = rating
    :param movie_dict: dict where key = (MovieID, CustomerID) and value = rating
    :param i: user id
    :param k: movie id
    :eturn: rating prediction
    """
    if i in user_dict.keys() and k in movie_dict.keys():
        prediction = predict_both_known(user_dict, movie_dict, i, k)
    elif i not in user_dict.keys() and k in movie_dict.keys():
        prediction = predict_unknown_user(user_dict, movie_dict, k)
    elif i in user_dict.keys() and k not in movie_dict.keys():
        prediction = predict_unknown_movie(user_dict, movie_dict, i)
    else:
        prediction = predict_both_unknown(user_dict)
    return prediction

def calculate_error(train_path, test_path, output):
    """Calculates testing error
    :param train_path: pathname for training data
    :param test_path: pathname for testing data
    :param output: output filename
    :return: (mean absolute error, root mean squared error)
    """
    (train_user_dict, train_movie_dict) = read_file(train_path)
    test_dict = read_testing_file(test_path)
    predictions = []
    with open(output, 'w') as file:
        writer = csv.writer(file)
        for key in test_dict.keys():
            movie = key[0]
            user = key[1]
            prediction = float(predict_rating(train_user_dict, train_movie_dict, user, movie))
            predictions.append(prediction)
            rating = float(test_dict.get((movie, user)))
            writer.writerow((movie, user, rating, round(prediction, 2)))
    ratings = np.asarray([float(i) for i in test_dict.values()])
    predictions = np.asarray(predictions)
    mae = np.sum(np.absolute(np.subtract(ratings, predictions)))/len(test_dict)
    rmse = np.sqrt(np.sum(np.subtract(ratings, predictions)**2)/len(test_dict))
    return(mae, rmse)
            
if __name__ == '__main__':
    args = parse_arguments()
    train = args['train'][0]
    test = args['test'][0]
    (mae, rmse) = calculate_error(train, test, 'predictions.txt')
    print("Mean Absolute Error: %s" % mae)
    print("Root Mean Square Error: %s" % rmse)