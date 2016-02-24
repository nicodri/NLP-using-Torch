
This task aims to build three different linear models for text classification from the Stanford Sentiment dataset (Socher et al., 2013).
* a Multinomial Naive Bayes which pretty fast to train
* a multinomial logistic regression
* a linear support vector machine.

We evaluate each of these models with their accuracy on the validation set. Our main work was then on tuning the hyperparameters.
The Stanford Sentiment dataset contains about 150 000 text reviews of movies with their rating (from 1 to 5), containing about 17 000 unique words. The reviews are already pre-processed and come as sparse bag-of-words features. The goal is to predict the rating of each review.
We used the Torch Lua framework to build the models and implemented them in the file HW1.lua. 
