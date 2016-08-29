# Natural Language Processing using Neural Networks

In this example the neural network is trained using positive and negative sentences and when new sentences are given as input to this network, the network predicts if the sentence is positive or negative.

-- features.py:
This file reads the data from two files (one with positive lines of text and another with negative lines of text) and builds lexicon and converts all the strings to numbers. 

-- featuresnn.py
This file contains the program for the neural network. The neural network is implemented using Tensorflow. The neural network used in this project has three hidden layers with 500, 100, 500 nodes in each hidden layer and one output layer with two nodes. 

In this sample two major concerns of data munging were <br />
* Data was not in numeric form.
* Data was not of uniform size.
 
Hence the major was to come with a solution to get over the above mentioned issues. The solution for this was to build a lexicon and then treat each sentence as a vector with respect to the lexicon. These vectors were sent to the neural netword for training. Whenever a new sentence had to be tested it was converted to a vector and was given as an input to the model.

Lexicon: A 1xN vector with all the unique words in the documents. Common words like "is, a, the, it, etc" were ignored while building the lexicon. All the words in the lexicon were in lemmatized form ie words like run, running, runs were considered as run. 

There were N (size of the lexicon) nodes in the input layer of the neural network. 

All sentences were represented in the form of a 1xN vector. 

For example: <br />
Lets consider the lexicon we built has the following words: [car, wall, driver, book, table] <br />

| Sentence      | Vector        |
| ------------- |:-------------:|
| Car hit the wall  | [1 1 0 0 0] |
| Book is on the table  | [0 0 0 1 1] |
| This book is about cars | [1 0 0 1 0] |

Similar all the sentences are transformed into vectors and are sent to the neural network for training and testing.

