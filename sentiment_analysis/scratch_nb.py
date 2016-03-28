# Naive Bayes From Scratch

from collections import Counter
import numpy as np

def create_word_count(train_directory):
    """Creates an rdd of word counts
    :param train_directory: directory of training data
    :return: rdd where key = word and value = count
    """
    train_rawdata = sc.wholeTextFiles(train_directory)
    
    train = train_rawdata.flatMap(lambda x: (x[1].lower().split(" "))).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)    
    return train

def calculate_denominator(training_neg, training_pos):
    """Calculates number of distinct words + 1 (denominator of naive bayes)
    :param training_neg: directory of training neg files
    :param training_pos: directory of training pos files
    :return: number of distinct words (int)
    """
    denominator = (training_neg.map(lambda x: x[0]).union(training_pos.map(lambda x: x[0]))).distinct().count() + 1
    return denominator
    
def calculate_prob(training, denominator, count):
    """Calculate probabilities for pos and neg files
    :param training: rdd of training data
    :param denominator: number of unique words 
    :param count: number of words in rdd
    :return: rdd of P(word|class) 
    """
    training_prob = training.map(lambda x: (x[0], np.log((x[1]+1)/float(denominator + count))))
    return training_prob

def classify(test_path, training_neg_prob, training_pos_prob, PRIOR_PROB, actual_class):
    """Reads and classifies all test files
    :param test_path: directory of test data
    :param training_neg_prob: rdd of P(word|neg)
    :param training_pos_prob: rdd of P(word|pos)
    :param PRIOR_PROB: constant
    :param actual_class: 0 or 1
    :return: rdd of classifications
    """
    test_rawdata = sc.wholeTextFiles(test_path)
    testing = test_rawdata.flatMapValues(lambda x: x.lower().split(" ")).map(lambda x: (x[1], x[0]))
    testing_neg_prob = testing.join(training_neg_prob).map(lambda x: x[1]).reduceByKey(lambda x, y: x + y + PRIOR_PROB)
    testing_pos_prob = testing.join(training_pos_prob).map(lambda x: x[1]).reduceByKey(lambda x, y: x + y + PRIOR_PROB)
    combined = testing_neg_prob.join(testing_pos_prob)
    compare = combined.map(lambda x: (actual_class, 1 if (x[1][0] < x[1][1]) else 0))
    return compare

def naive_bayes_scratch(train_neg_directory, train_pos_directory, test_neg_directory, test_pos_directory):
    """Fit naive bayes classifier to predict sentiment polarity
    :param train_neg_directory: directory of training neg files
    :param train_pos_directory: directory of training pos files
    :param test_neg_directory: directory of testing neg files
    :param test_pos_directory: directory of testing pos files
    :return: accuracy of classifier 
    """
    
    # Create rdd of the words and their counts
    training_neg = create_word_count(train_neg_directory)
    training_pos = create_word_count(train_pos_directory)
    
    # Calculates denominator of naive bayes formula
    denominator = calculate_denominator(training_neg, training_pos)
    
    # Counts number of words in the neg train and pos train
    neg_count = training_neg.count()
    pos_count = training_pos.count()
    
    # Calculates P(w|neg) and P(w|pos)
    training_neg_prob = calculate_prob(training_neg, denominator, neg_count)
    training_pos_prob = calculate_prob(training_pos, denominator, pos_count)
    
    PRIOR_PROB = np.log(0.5)
    
    # Classifies pos and neg test files
    testing_pos = classify(test_pos_directory, training_neg_prob, training_pos_prob, PRIOR_PROB, 1)
    testing_neg = classify(test_neg_directory, training_neg_prob, training_pos_prob, PRIOR_PROB, 0)
    
    # Joins the pos and neg test classifications
    predictionAndLabel = testing_pos.union(testing_neg)
    
    # Calculates accuracy 
    accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / predictionAndLabel.count()
    
    return accuracy

if __name__ == '__main__':
    #Configure spark with your S3 access keys
    AWS_ACCESS_KEY_ID = "ACCESS_KEY_ID"
    AWS_SECRET_ACCESS_KEY = "SECRET_ACCESS_KEY"

    sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
    sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)
    
    train_pos_directory = "s3n://bucket_name/aclImdb/aclImdb/train/pos/*.txt"
    train_neg_directory = "s3n://bucket_name/aclImdb/aclImdb/train/neg/*.txt"
    test_pos_directory = "s3n://bucket_name/aclImdb/aclImdb/test/pos/*.txt"
    test_neg_directory = "s3n://bucket_name/aclImdb/aclImdb/test/neg/*.txt"

    PRIOR_PROB = np.log(0.5)
    
    accuracy = naive_bayes_scratch(train_neg_directory, train_pos_directory, test_neg_directory, test_pos_directory)