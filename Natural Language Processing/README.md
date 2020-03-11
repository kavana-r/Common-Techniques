# Natural Language Processing ( Text Analytics ) 

#### This repo strictly focuses on static NLP analytics to gather insights. 

#### Additionally, material below is to get background on concepts prior to application. Please refer to Application folder to find code.

NLP is a branch of artificial intelligence that deals with analyzing, understanding and generating the languages that humans use naturally in order to interface with computers in both written and spoken contexts using natural human languages instead of computer languages.

#### Applications of NLP
    Machine translation(Google Translate)
    Natural language generation
    Web Search
    Spam filters
    Sentiment Analysis
    Topic Detection/Modelling
    Chatbots
    … and many more


# Steps: 
1. Data Acquisition
2. Data/Text Pre Processing
3. Data Trasnformation/Modelling
4. Model Evaluation


![How to Get started]( https://github.comcast.com/storage/user/18009/files/20056280-61ff-11ea-9978-3729211220ef)


### Characteristics of Text data 
Corpus & Corpa:
![Units of Text data: Corpora]( https://github.comcast.com/storage/user/18009/files/f7806700-6204-11ea-82ed-df04f95a9923)

## Data/Text Pre Processing
Prior to modelling, we need to get the text data ready in an easily digestable format for the computer. Few of the steps involved in doing so: 

    a. Convert all characters to lower case

    b. Remove punctuations,numbers,and all other symbols that are not letters of the alphabet or dot

    c. Remove extra white spaces, escape-tables,escape-newlines, trailing & leading spaces

    d. Stem words: Reduces words to their root. Shrinks the total # of unique words in the data which reduces the noise in  data along with helping out in dimensionality

    e. Remove Stop Words

    f. Tokenize
![PreProcessing Simple Example]( https://github.comcast.com/storage/user/18009/files/4d183c00-6224-11ea-9c85-36c975ff7455)

## Data Tranformation/Modelling: 


### Frequency Based Approaches:
---
### Topic Modelling using Bag of Words + TF-IDF

    Basic "Bag-of-words" Approach:

    1. A document is treated as a bag of words where word position and structure do not matter

    2. Text is cleaned until only stripped down word-roots remain

    3. Each occurrence of a word is then counted in each document

    4. Word frequencies are recorded and arranged into a matrix of words by documents, additional weighting may be applied

    5. The numeric representation of your text corpus is this matrix

    6. Document similarity is based on a similar words appear in documents with similar meaning

**Process: Text Pre Processing --> Document-Term Matrix (DTM) --> Weighted TF-IDF Weights --> LDA Topic Modelling**

**Document-Term-Matrix (DTM)** : 

A matrix of unique words counted in each document. Corpus vocbulary consists of all of the unique terms (ie. column names of DTM) and their total counts across all documents (ie. column sums)

![DTM]( https://github.comcast.com/storage/user/18009/files/acc31700-6225-11ea-8971-2e73c12d91dc)

A Typical Word distribution in language & Corpus: 
![Word Dist]( https://github.comcast.com/storage/user/18009/files/54404980-6226-11ea-85c6-4a0186899e7c)

As we see above, a typical corpus is highly skewed right. This is due to natural prevelance of some words due to English as a language. The most used are typically stop words like "and","Along" etc. The lower end of the tail refers to extremly sparse words that

**How do you take care of this ?**

As you guessed, a version of assigned weighting (data transformation). 

Infact, it has been proven by H.P. Luhn (a researcher for IBM) in 1958 that the power of significant words is approximately
normally distributed and the top of the bell-shaped curve coincides with the mid-portion of the word frequency distribution as seen below

![Normal distribution of Significant words]( https://github.comcast.com/storage/user/18009/files/ff510300-6226-11ea-8359-42f1a586bae9)

Essentially, getting rid of most frequent and extremly sparse words helps us with dimensionality reduction and prevent overfitting.

**TF-IDF: Inverse Document Frequency**

Term Frequency-Inverse Document Frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.

![TF-IDF]( https://github.comcast.com/storage/user/18009/files/6a9ad500-6227-11ea-9f6d-7f16c2a93733)

![Example of DTM to DTM w. TF-IDF Weights]( https://github.comcast.com/storage/user/18009/files/a33aae80-6227-11ea-93f3-9b28ae4c0860)

**So what ? What can we do with this ?**

Variations of the tf-idf weighting scheme are often used by search engines in scoring and ranking a document's relevance given a query.

Major use case is calculating similarity:
1. Information Retreival
2. Similarity

**Example on similarity use case:**

![Similarity](https://github.comcast.com/storage/user/18009/files/28d27180-6382-11ea-9400-bf50ab90f37c)


**Short comings of bag of words approach**
    
    1. Dimensionality problem beacuse the total dimension is the vocabulary size and it can easily over-fit your model.

    2. Bag of words representation doesn’t consider the semantic relation between words, it just focus on count of word and neglect the arrangement, n-grams and tagging in sentence.

---
### LDA: Latent Discriminant Analysis
LDA: form of unsupervised learning that views documents as "bag of words". 

In fact, Blei (who developed LDA), points out in the introduction of the paper of 2003 (entitled "Latent Dirichlet Allocation") that LDA addresses the shortcomings of the TF-IDF model and leaves this approach behind. LDA is a probabilistic model that tries to estimate probability distributions for topics in documents and words in topics. The weighting of TF-IDF is not necessary for this.

However, alot of resources say that LDA can be performed on both Bag of words vectors OR TF-IDF matrix. Please refer to documentation for the package you decided to use (Gensim most common).

After pre processing of data, we can easily run topic modelling with LDA using gensim & pyLDAvis. This version is on bag of words vector. Below is the output for rNPS Comments:

![pyLDAvis Topic Modelling on rNPS Comments Visualization]( https://github.comcast.com/storage/user/18009/files/88b01600-62b7-11ea-8235-d0ff8644f5b0)

---
### N-Grams:

In the fields of computational linguistics and probability, an n-gram is a contiguous sequence of n items from a given sample of text or speech. So essentially a sequence of phrases based on window size. Ex. n = 2 then High Price and w=3 Ridiculous high price

n (window size) = 2 ..Bigram
n (window size) = 3 ..Trigram

As you can guess, this is quite useful in:
- Understanding key themes/topics...especially on action based dataset such as BOT CHAT DATA
- Auto completion of sentences, auto spell check

Side note: For auto-completion, we would need to essentially calculate the probability of a word occuring after a certain word. Ex. whats the probability of the word "like" occuring after the word "really":

![N Gram Probability Calculation]( https://github.comcast.com/storage/user/18009/files/f9652b80-62d3-11ea-9860-ce08ffffe4f5)

---
### ML Based Approaches: 

As you can see frequency based methods are great for quick hits and on a more action based datasets. However, they fail to capture Semantics/Context of the word and ignores order. To solve for this, we can create unique vectors for each word in a n dimensional space. The vector creation takes into similarity between words along with order.

### Skip Gram Model: Word2Vec  

![Basics of Skip Gram](https://github.comcast.com/storage/user/18009/files/d38c5680-62d4-11ea-92f2-6d9cd3400fb2)

#### A quick introduction on Neural Networks and its application for NLP. 

I don't aim to go into all the intricaies of neural networks. However, the slide below explains neural networks at its most basic and why they are important in NLP.

![Basics of Neural Networks in NLP]( https://github.comcast.com/storage/user/18009/files/ab045c80-62d4-11ea-86d0-228ab0d5a8d8)

Word2Vec uses a trick you may have seen elsewhere in machine learning. We’re going to train a simple neural network with a single hidden layer to perform a certain task, but then we’re not actually going to use that neural network for the task we trained it on! Instead, the goal is actually just to learn the weights of the hidden layer–we’ll see that these weights are actually the “word vectors” that we’re trying to learn.

**Fake Task**:

We’re going to train the neural network to do the following. Given a specific word in the middle of a sentence (the input word), look at the words nearby and pick one at random. The network is going to tell us the probability for every word in our vocabulary of being the “nearby word” that we chose.

The output probabilities are going to relate to how likely it is find each vocabulary word nearby our input word

We’ll train the neural network to do this by feeding it word pairs found in our training documents. The below example shows some of the training samples (word pairs) we would take from the sentence “The quick brown fox jumps over the lazy dog.” I’ve used a small window size of 2 just for the example. The word highlighted in blue is the input word.

![Skips](https://github.comcast.com/storage/user/18009/files/e0ff0c00-62e7-11ea-9268-15339e77cbf1)


The network is going to learn the statistics from the number of times each pairing shows up. 

**Model Details**

So how is this all represented?

First of all, you know you can’t feed a word just as a text string to a neural network, so we need a way to represent the words to the network. To do this, we first build a vocabulary of words from our training documents–let’s say we have a vocabulary of 10,000 unique words.

We’re going to represent an input word like “ants” as a one-hot vector. This vector will have 10,000 components (one for every word in our vocabulary) and we’ll place a “1” in the position corresponding to the word “ants”, and 0s in all of the other positions.

The output of the network is a single vector (also with 10,000 components) containing, for every word in our vocabulary, the probability that a randomly selected nearby word is that vocabulary word.

Here’s the architecture of our neural network.
![Neural Network Architecture](https://github.comcast.com/storage/user/18009/files/de051b00-62e9-11ea-8632-c072fa3a5cfc)

When training this network on word pairs, the input is a one-hot vector representing the input word and the training output is also a one-hot vector representing the output word. But when you evaluate the trained network on an input word, the output vector will actually be a probability distribution (i.e., a bunch of floating point values, not a one-hot vector).

**What is a hidden layer ?**

You can think of layers in a neural network as container of neurons. A layer groups a number of neurons together. It is hold for holding information.

Hidden layer reside in between input and output layers and this is the primary reason why they are referred to as hidden.
For our example, we’re going to say that we’re learning word vectors with 300 features. So the hidden layer is going to be represented by a weight matrix with 10,000 rows (one for every word in our vocabulary) and 300 columns (one for every hidden neuron).

![Hidden Layer](https://github.comcast.com/storage/user/18009/files/7a77fd80-637c-11ea-8c64-efc666a67042)

So the end goal of all of this is really just to learn this hidden layer weight matrix – the output layer we’ll just toss when we’re done!

![Example](https://github.comcast.com/storage/user/18009/files/a5625180-637c-11ea-907c-5e6d0355ecee)

This means that the hidden layer of this model is really just operating as a lookup table. The output of the hidden layer is just the “word vector” for the input word.


**What is a output layer ?**
The output layer is a softmax regression classifier. Softmax regression (or multinomial logistic regression) is a generalization of logistic regression to the case where we want to handle multiple classes. 

For deeper understanding, please refer to standford resource below:
http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/

Each output neuron (one per word in our vocabulary!) will produce an output between 0 and 1, and the sum of all these output values will add up to 1.

Specifically, each output neuron has a weight vector which it multiplies against the word vector from the hidden layer, then it applies the function exp(x) to the result. Finally, in order to get the outputs to sum up to 1, we divide this result by the sum of the results from all 10,000 output nodes.

![Example](https://github.comcast.com/storage/user/18009/files/e65a6600-637c-11ea-94fd-4951c4498c52)

**Optimizing your skip-gram model**

Negative Sampling:
http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/

**Implementation Example**

As we discussed above, the output of a skip gram neural network is a weight vector with probability of one word being close to another in our vocabulary. 

Leveraging neural networks ability to understand similar word, we can run a unsupervised learning model such as K Means to cluster similar words together. Below is my take on 1 year of rNPS comments:

![Skip Gram K Means Result](https://github.comcast.com/storage/user/18009/files/5965db80-6381-11ea-85b2-0db3884dae7e)

## Model Comparision: Bag of Words vs Skip Gram vs ELMO
![Model Comparision](https://github.comcast.com/storage/user/18009/files/f74f9c80-62d4-11ea-9e14-cb70d10b0ab2)

## Topic Detection Overview: Passs I on rNPS/tNPS Commentary 
![Topic Detection Overview]( https://github.comcast.com/storage/user/18009/files/65765f80-6200-11ea-8908-b1695af8c103)
