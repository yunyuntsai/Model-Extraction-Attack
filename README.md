# Model-Extraction-Attack
Online model extraction attacks against ML service. 
Extract A model trained by ourselves on AWS, to which we only get black-box access. 
Our attacks only use exposed APIs, re-construct the model by issuing queries with a single feature specified, such as to obtain equations with a single unknown in X .
For models with multiple classes,  use c · (d + 1) queries where c is the # of classes and d is the # of features
For example, the adult dataset which we want to predict the Target "race", it has 5 classes and 113 features. Hence, we need 5 * (113+1) queries to reverse the unknown features.
Using line searches to reverse-engineer the binning transformation
