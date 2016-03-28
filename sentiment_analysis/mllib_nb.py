# MLLib Naive Bayes Classifer to Predict Sentiment Polarity
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes

def label_points(pos, neg):
    """Preprocess data for naive bayes
    :param pos: directory path for positive files
    :param neg: directory path for negative files
    :return: rdd of full labeled dataset
    """
    tf_pos = HashingTF().transform(
        pos.map(lambda doc: doc[1].lower().split(' '), preservesPartitioning=True))
    
    tf_neg = HashingTF().transform(
        neg.map(lambda doc: doc[1].lower().split(' '), preservesPartitioning=True))
    
    tf_pos_label = tf_pos.map(lambda x: LabeledPoint(1, x))
    
    tf_neg_label = tf_neg.map(lambda x: LabeledPoint(0, x))
    
    combined = tf_pos_label.union(tf_neg_label)
    
    return combined
    
def naive_bayes(train_pos, train_neg, test_pos, test_neg):
    """Fit naive bayes classifier to predict sentiment polarity 
    :param train_pos: directory path for positive training files
    :param train_neg: directory path for negative training files
    :param test_pos: directory path for positive testing files
    :param test_neg: directory path for negative training files
    :return: accuracy of classifier
    """
    training = label_points(train_pos, train_neg)
    
    test = label_points(test_pos, test_neg)
    
    model = NaiveBayes.train(training)
    
    predictionAndLabel = test.map(lambda p : (model.predict(p.features), p.label))
    
    accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / test.count()
    
    return accuracy

if __name__ == '__main__': 
    train_pos_directory = "/Users/Documents/aclImdb/train/pos/*.txt"
    train_neg_directory = "/Users/Documents/aclImdb/train/neg/*.txt"
    test_pos_directory = "/Users/Documents/aclImdb/test/pos/*.txt"
    test_neg_directory = "/Users/Documents/aclImdb/test/neg/*.txt"

    # Load files
    train_pos = sc.wholeTextFiles(train_pos_directory)
    train_neg = sc.wholeTextFiles(train_neg_directory)
    test_pos = sc.wholeTextFiles(test_pos_directory)
    test_neg = sc.wholeTextFiles(test_neg_directory)
       
    naive_bayes(train_pos, train_neg, test_pos, test_neg)