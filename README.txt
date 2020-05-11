CS 4641 Final Project
---------------------
Sunny Qi


Data Information:
- Downloaded from: http://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition as csv.
- Provided as "seizuredata.csv"
- Cleaning/Modifications (done in get_test_train() function in code):
	- replaced class labels of 3-5 with 0. Class label of 1 means there was seizure activity recorded by the EEG. Class label of 2 meant the EEG was recording an area of the brain with a tumor. The other classes 3-5 recorded no seizure activity and did not directly record the area of a tumor, if a tumor was present at all. So I decided to focus only on classifying the presence of medical problems as recorded by an EEG (tumors and seizures).
	- One data point was missing a class label and so it had a class label of -1, I removed this data point from the set.

Libraries (installed with pip):
- sklearn
- pprint
- matplotlib
- numpy
- pandas

Instructions for Running:
- Command has the form python3.7 project.py <classifier> <task> where the classifier is one of: 'rf', 'svm', 'nn'
	- When running with classifier 'rf', the options for the <task> are:
		-'basic' - which runs the basic random forest classifier from sci-kit learn with default parameters
		-'random' - which runs RandomSearchCV for 50 iterations and produces a table of the top results
		-'grid' - which runs GridSearchCV for 2 parameteres and produces a relevant graph of the results
		-'cv' - which runs 5-fold cross validation using the best parameters found in the grid search, and the full dataset.
	- When running with classifier 'svm', the options for the <task> are:
		-'basic' - which runs the basic support vector classifier from sci-kit learn with default parameters
		-'linear' - which runs the linear SVC from sci-kit learn with default parameters but increased max iterations
		-'random' - which runs RandomSearchCV for 50 iterations and produces a table of the top results
		-'grid' - which runs GridSearchCV for 2 parameteres and produces a relevant graph of the results
		-'cv' - which runs 5-fold cross validation using the best parameters found in the grid search, and the full dataset.
	- When running with classifier 'nn', the options for the <task> are:
		-'basic' - which runs the basic MLPClassifier from sci-kit learn with 2 hidden layers and default parameters
		-'random' - which runs RandomSearchCV for 50 iterations and produces a table of the top results
		-'grid' - which runs GridSearchCV for 2 parameteres and produces a relevant graph of the results
		-'cv' - which runs 5-fold cross validation using the best parameters found in the grid search, and the full dataset.