#importing the required modules.
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

#defining the lemmatizer. 
lemmatizer = WordNetLemmatizer()
#defining the number of lines in the document to read.
lines = 10000

#function to create a lexicon.
#the function takes two strings (file names of positive and 
#negative files) 
def create_lexicon(pos, neg):
	#creating an empty list for temporary lexicon.
	lexicon = []
	#parsing through the files line by line and building a lexicon
	for i in [pos, neg]:
		with open(i, 'r') as f:
			contents = f.readlines()
			for l in contents[:lines]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)
	#lemmatizing the words in lexicon. 
	#and getting rid of very common and uncommon words.
	lexicon = [lemmatizer.lemmatize(j) for j in lexicon]
	wc = Counter(lexicon)
	#list for adding the required words.
	l2 = []
	#loop getting rid of common words.
	for w in wc:
		if 1000 > wc[w] > 50:
			l2.append(w)
	print(len(l2))
	return l2

#building a feature set with words in the lexicon.
#this function converts the strings to numbers. 
def sample_handling(sample, lexicon, classification):
	#defining an empty list for the feature set.
	fs = []
	#building the feature set. 
	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents[:lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			#defining an empty feature set. A numpy array. 
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			features = list(features)
			fs.append([features, classification]) 
	return fs

#creating the testing and training data. 
#positive == [1,0]
#negative == [0,1]
def create_labels(pos, neg, test_size=0.1):
	#creating the lexicon.
	lexicon = create_lexicon(pos, neg)
	#creating the feature sets with positive and negative data. 
	features = []
	features += sample_handling('pos.txt',lexicon, [1,0])
	features += sample_handling('neg.txt',lexicon, [0,1])
	#shuffling the features before sending it to the NN.
	random.shuffle(features)
	#converting the list to a numpy array.
	features = np.array(features)
	#calculating the test data size.	
	testing_size = int(test_size*len(features))
	#creating the training data.
	trainx = list(features[:,0][:-testing_size])
	trainy = list(features[:,1][:-testing_size])
	#creating the testing data.
	testx = list(features[:,0][-testing_size:])
	testy = list(features[:,1][-testing_size:])
	#returning the datasets.
	return trainx, trainy, testx, testy

#checking for main.
if __name__ == '__main__':
	trainx, trainy, testx, testy = create_labels('pos.txt', 'neg.txt')
	#saving the data to a pickle.
	with open('sets.pickle', 'wb') as f:
		pickle.dump([trainx, trainy, testx, testy], f)
	





