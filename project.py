import matplotlib.pyplot as plt
import numpy
import pandas as pd
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
import sys
import time

def get_test_train():
	data = numpy.genfromtxt('seizuredata.csv',delimiter=',',dtype=int)
	# delete data points that are missing a class label (only 1)
	data = numpy.delete(data, numpy.where(data[:, -1] < 0), axis=0)
	data_X = data[:,:-1].astype(int)
	data_Y = data[:,-1].reshape(-1,)
	numpy.random.seed(1567708903)
	shuffled_idx = numpy.random.permutation(data.shape[0])
	cutoff = int(data.shape[0]*0.8)
	train_data = data[shuffled_idx[:cutoff]]
	test_data = data[shuffled_idx[cutoff:]]
	train_X = train_data[:,:-1].astype(int)
	train_Y = train_data[:,-1].reshape(-1,)
	test_X = test_data[:,:-1].astype(int)
	test_Y = test_data[:,-1].reshape(-1,)

	# We are only looking at whether the patient had a seizure, or the EEG was recording a tumor or neither, so replace all other classes with 0 for neither
	data_Y[data_Y > 2] = 0
	train_Y[train_Y > 2] = 0
	test_Y[test_Y > 2] = 0
	return train_X, train_Y, test_X, test_Y, data_X, data_Y

def random_search_random_forest(train_X, train_Y, test_X, test_Y):
	rf = RandomForestClassifier(bootstrap=True)
	max_depth = [i for i in range(10,300)]
	max_depth.append(None)
	n_estimators = [i for i in range(1,1000)]
	min_samples_split = [i for i in range(2,30)]
	min_samples_leaf = [i for i in range(1,30)]
	max_samples = [i/10.0 for i in range(1,10)]
	max_samples.append(None)
	random_grid = {'n_estimators': n_estimators,
				'max_depth': max_depth,
				'max_features': ['sqrt', 'log2', None],
				'min_samples_split': min_samples_split,
				'min_samples_leaf': min_samples_leaf,
				'max_samples': max_samples}
	rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=50, verbose=1, random_state=10, n_jobs=-1, cv=3)
	rf_random.fit(train_X, train_Y)
	# plot table
	results = pd.DataFrame(rf_random.cv_results_)
	top_results = results[results['rank_test_score'] < 6.0]
	top_results = top_results[['mean_test_score', 'std_test_score', 'rank_test_score', 'param_n_estimators', 'param_max_depth', 'param_max_samples']]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.axis('off')
	table = ax.table(cellText=top_results.values,colLabels=top_results.columns, loc='center')
	plt.savefig('RandomForestRandomSearchTable.png', dpi=300)
	print('Random Search Random Forest Best Score:', rf_random.best_estimator_.score(test_X, test_Y))
	print('Best Params:')
	pprint(rf_random.best_params_)

def grid_search_random_forest(train_X, train_Y, test_X, test_Y):
	rf = RandomForestClassifier()
	param_grid = {'n_estimators': [i for i in range(90,121)],
				'max_features': ['sqrt', 'log2', None]}
	grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1, verbose=1, cv=3)
	grid_search.fit(train_X, train_Y)
	# plot table
	results = pd.DataFrame(grid_search.cv_results_)
	results = results[['mean_test_score', 'param_n_estimators', 'param_max_features']]
	results.fillna(value='None', inplace=True)
	fig, ax = plt.subplots()
	for d in results['param_max_features'].unique():
		data = results[results['param_max_features'] == d]
		data.plot(x='param_n_estimators', y='mean_test_score', ax=ax, label=d)
	plt.title('Random Forest Grid Search Scores')
	plt.savefig('RandomForestGridSearch.png')
	print('Grid Search Random Forest Best Score:', grid_search.best_estimator_.score(test_X, test_Y))
	print('Best Params:')
	pprint(grid_search.best_params_)

def cross_validate_random_forest(data_X, data_Y):
	rf = RandomForestClassifier(max_depth=None,
								min_samples_leaf=1,
								min_samples_split=2,
								max_features='sqrt',
								max_samples=None,
								n_estimators=110)
	scores = cross_val_score(rf, data_X, data_Y, cv=5, verbose=1, n_jobs=-1)
	print('Mean:', scores.mean())
	print('Standard Deviation:', scores.std())


def random_search_support_vector(train_X, train_Y, test_X, test_Y):
	svm = SVC()
	C = [i for i in range(1, 100)]
	C.extend([i/10.0 for i in range(1,10)])
	C.extend([i/100.0 for i in range(1,10)])
	gamma = [i for i in range(1, 10)]
	gamma.extend([i/10.0 for i in range(1,10)])
	gamma.extend([i/100.0 for i in range(1,10)])
	gamma.extend([i/1000.0 for i in range(1,10)])
	gamma.extend(['scale', 'auto'])
	random_grid = {'C': C,
				'gamma': gamma,
				'kernel': ['rbf', 'poly', 'sigmoid'],
				'degree': [i for i in range(2, 5)]}
	svm_random = RandomizedSearchCV(estimator=svm, param_distributions=random_grid, n_iter=50, verbose=1, random_state=10, n_jobs=-1, cv=3)
	svm_random.fit(train_X, train_Y)
	# plot table
	results = pd.DataFrame(svm_random.cv_results_)
	top_results = results[results['rank_test_score'] < 6.0]
	top_results = top_results[['mean_test_score', 'std_test_score', 'rank_test_score','param_C', 'param_gamma', 'param_kernel']]
	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.axis('off')
	table = ax.table(cellText=top_results.values,colLabels=top_results.columns, loc='center')
	plt.savefig('SVMRandomSearchTable.png', dpi=300)
	print('Random Search SVM Best Score:', svm_random.best_estimator_.score(test_X, test_Y))
	print('Best Params:')
	pprint(svm_random.best_params_)

def grid_search_support_vector(train_X, train_Y, test_X, test_Y):
	svm = SVC(kernel='rbf')
	C = [i for i in range(13, 33)]
	gamma = ['scale', 'auto', .01, .001]
	param_grid = {'C': C,'gamma': gamma}
	grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, n_jobs=-1, verbose=1, cv=3)
	grid_search.fit(train_X, train_Y)
	# plot table
	results = pd.DataFrame(grid_search.cv_results_)
	results = results[['mean_test_score', 'param_C', 'param_gamma']]
	results.to_csv('test.csv')
	fig, ax = plt.subplots()
	for d in results['param_gamma'].unique():
		data = results[results['param_gamma'] == d]
		data.plot(x='param_C', y='mean_test_score', ax=ax, label=d)
	plt.title('SVM Grid Search Scores')
	plt.savefig('SVMGridSearch.png')
	print('Grid Search SVM Best Score:', grid_search.best_estimator_.score(test_X, test_Y))
	print('Best Params:')
	pprint(grid_search.best_params_)

def cross_validate_support_vector(data_X, data_Y):
	svm = SVC(kernel='rbf', gamma='scale', C=22)
	scores = cross_val_score(svm, data_X, data_Y, cv=5, verbose=1, n_jobs=-1)
	print('Mean:', scores.mean())
	print('Standard Deviation:', scores.std())

def random_search_neural_network(train_X, train_Y, test_X, test_Y):
	nn = MLPClassifier()
	hidden_layer_sizes = [(i,j) for i in range(30,500,10) for j in range(30,500,10)]
	hidden_layer_sizes.extend([(i,j,k) for i in range(30,500,10) for j in range(30,500,10) for k in range(30,500,10)])
	hidden_layer_sizes.extend([(i,j,k,l) for i in range(30,500,10) for j in range(30,500,10) for k in range(30,500,10) for l in range(30,500,10)])
	alpha = [i/100.0 for i in range(1,10)]
	alpha.extend([i/1000.0 for i in range(1,10)])
	random_grid = {'hidden_layer_sizes': hidden_layer_sizes,
				'activation': ['identity', 'logistic', 'tanh', 'relu'],
				'solver': ['lbfgs', 'sgd', 'adam'],
				'alpha': alpha,
				'learning_rate': ['constant', 'invscaling', 'adaptive'],
				'learning_rate_init': alpha}
	nn_random = RandomizedSearchCV(estimator=nn, param_distributions=random_grid, n_iter=50, verbose=1, random_state=10, n_jobs=-1, cv=3, return_train_score=True)
	nn_random.fit(train_X, train_Y)
	# plot table
	results = pd.DataFrame(nn_random.cv_results_)
	top_results = results[results['rank_test_score'] < 6.0]
	top_results = top_results[['mean_test_score', 'std_test_score', 'rank_test_score', 'param_hidden_layer_sizes','param_activation', 'param_solver', 'param_alpha']]
	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.axis('off')
	table = ax.table(cellText=top_results.values,colLabels=top_results.columns, loc='center')
	plt.savefig('NeuralNetRandomSearchTable.png', dpi=300)
	print('Random Search Neural Network Best Score:', nn_random.best_estimator_.score(test_X, test_Y))
	print('Best Params:')
	pprint(nn_random.best_params_)

def grid_search_neural_network(train_X, train_Y, test_X, test_Y):
	nn = MLPClassifier(activation='relu', solver='adam', learning_rate_init=0.006)
	param_grid = {'hidden_layer_sizes': [(i,j,k,l) for i in range(227,234,3) for j in range(307,314,3) for k in range(427,434,3) for l in range(137,144,3)],
				'alpha': [i/1000.0 for i in range(5,8)]}
	grid_search = GridSearchCV(estimator=nn, param_grid=param_grid, n_jobs=-1, verbose=1, cv=3)
	grid_search.fit(train_X, train_Y)
	# plot table
	results = pd.DataFrame(grid_search.cv_results_)
	results = results[['mean_test_score', 'param_hidden_layer_sizes', 'param_alpha', 'rank_test_score']]
	top_results = results[results['rank_test_score'] < 6.0]
	fig=plt.figure()
	ax = fig.add_subplot(111)
	ax.axis('off')
	table = ax.table(cellText=top_results.values,colLabels=top_results.columns, loc='center')
	plt.savefig('NeuralNetGridSearchTable.png', dpi=300)
	plt.clf()
	plt.scatter(results.param_alpha, results.mean_test_score)
	plt.xlabel('Alpha')
	plt.ylabel('Mean Score')
	plt.title('Neural Network Grid Search Scores')
	plt.savefig('NNGridSearch.png')
	print('Grid Search Neural Network Best Score:', grid_search.best_estimator_.score(test_X, test_Y))
	print('Best Params:')
	pprint(grid_search.best_params_)

def cross_validate_neural_network(data_X, data_Y):
	nn = MLPClassifier(activation='relu', solver='adam', learning_rate_init=0.006, alpha=.007, hidden_layer_sizes=(230, 313, 430, 143))
	scores = cross_val_score(nn, data_X, data_Y, cv=5, verbose=1, n_jobs=-1)
	print('Mean:', scores.mean())
	print('Standard Deviation:', scores.std())

def main():
	start = time.time()
	if len(sys.argv) == 3:
		classifier = sys.argv[1]
		task = sys.argv[2]
		train_X, train_Y, test_X, test_Y, data_X, data_Y = get_test_train();
		# To see how many of each label are in training data
		# unique, counts = numpy.unique(train_Y, return_counts=True)
		# print(dict(zip(unique, counts)))
		# unique, counts = numpy.unique(test_Y, return_counts=True)
		# print(dict(zip(unique, counts)))
		if classifier == 'rf':
			if task == 'basic':
				rf = RandomForestClassifier()
				rf.fit(train_X, train_Y)
				print('Random Forest (Bagging) Default Score:', rf.score(test_X, test_Y))
				end = time.time()
				print('Elapsed Time:', (end - start)/60.0, 'minutes')
			elif task == 'random':
				random_search_random_forest(train_X, train_Y, test_X, test_Y)
				end = time.time()
				print('Elapsed Time:', (end - start)/60.0, 'minutes')
			elif task == 'grid':
				grid_search_random_forest(train_X, train_Y, test_X, test_Y)
				end = time.time()
				print('Elapsed Time:', (end - start)/60.0, 'minutes')
			elif task == 'cv':
				cross_validate_random_forest(data_X, data_Y)
				end = time.time()
				print('Elapsed Time:', (end - start)/60.0, 'minutes')
			else:
				print('Unrecognized Task Specified. Valid Options for Random Forest Are: \'basic\', \'random\', \'grid\', \'cv\'.')
		elif classifier == 'svm':
			if task == 'basic':
				svc = SVC()
				svc.fit(train_X, train_Y)
				print('SVM (RBF Kernel) Score:', svc.score(test_X, test_Y))
				end = time.time()
				print('Elapsed Time:', (end - start)/60.0, 'minutes')
			elif task == 'linear':
				svc = LinearSVC(tol=.001, max_iter=100000)
				svc.fit(train_X, train_Y)
				print('SVM (Linear Kernel) Score:', svc.score(test_X, test_Y))
				end = time.time()
				print('Elapsed Time:', (end - start)/60.0, 'minutes')
			elif task == 'random':
				random_search_support_vector(train_X, train_Y, test_X, test_Y)
				end = time.time()
				print('Elapsed Time:', (end - start)/60.0, 'minutes')
			elif task == 'grid':
				grid_search_support_vector(train_X, train_Y, test_X, test_Y)
				end = time.time()
				print('Elapsed Time:', (end - start)/60.0, 'minutes')
			elif task == 'cv':
				cross_validate_support_vector(data_X, data_Y)
				end = time.time()
				print('Elapsed Time:', (end - start)/60.0, 'minutes')
			else:
				print('Unrecognized Task Specified. Valid Options for Support Vector Machines Are: \'basic\', \'linear\', \'random\', \'grid\', \'cv\'.')
		elif classifier == 'nn':
			if task == "basic":
				nn = MLPClassifier(hidden_layer_sizes=(100,100))
				nn.fit(train_X, train_Y)
				print('Neural Network (2 Hidden Layers) Score: ', nn.score(test_X, test_Y))
				end = time.time()
				print('Elapsed Time:', (end - start)/60.0, 'minutes')
			elif task == 'random':
				random_search_neural_network(train_X, train_Y, test_X, test_Y)
				end = time.time()
				print('Elapsed Time:', (end - start)/60.0, 'minutes')
			elif task == 'grid':
				grid_search_neural_network(train_X, train_Y, test_X, test_Y)
				end = time.time()
				print('Elapsed Time:', (end - start)/60.0, 'minutes')
			elif task == 'cv':
				cross_validate_neural_network(data_X, data_Y)
				end = time.time()
				print('Elapsed Time:', (end - start)/60.0, 'minutes')
			else:
				print('Unrecognized Task Specified. Valid Options for Neural Networks Are: \'basic\', \'random\', \'grid\', \'cv\'.')
		else:
			print('Unrecognized Classifier Specified. Valid Options are: \'rf\', \'svm\', \'nn\'')
	else:
		print('Please specify a classifier and a task.')

if __name__ == '__main__':
	main()