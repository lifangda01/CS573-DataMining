from preprocess import *
from dt import DecisionTree, BaggedDecisionTrees, RandomForest, BoostedDecisionTrees
from svm import SupportVectorMachine
from sklearn.tree import DecisionTreeClassifier as skDecisionTree
from sklearn.ensemble import RandomForestClassifier as skRandomForest

def cross_validate(csv_file_name, losses_file_name, models,
					tssp, num_words, max_depth, n_estimators,
					debug=False):
	'''
		Perform 10-fold incremental cross validation.
	'''
	total_num = 2000
	lists_of_dict = []
	setups = [(p,w,d,t) for p in tssp for w in num_words for d in max_depth for t in n_estimators]
	losses = zeros((len(models), len(setups), 10)) # #models, #cases, #folds
	sklosses = zeros((2, len(setups), 10))
	generate_train_and_test_files_cv(csv_file_name, 10)
	# Generate temp CV files
	for i in range(10):
		lists_of_dict.append( csv_to_dict('cv%d.dat'%(i)) )
	i = 0
	for prop, nwords, maxdep, ntrees in setups:
		for j in range(10):
			# Contruct train set
			training_lists_of_dict = lists_of_dict[:j] + lists_of_dict[j+1:]
			training_list_of_dict = [item for sublist in training_lists_of_dict for item in sublist]
			testing_list_of_dict = lists_of_dict[j]
			# Randomly select samples
			random_indices = permutation(len(training_list_of_dict))
			random_indices = random_indices[:int(total_num*prop)]
			training_list_of_dict = [training_list_of_dict[k] for k in random_indices]
			# Find the word features
			feature_words = construct_word_feature(training_list_of_dict, nwords)
			# Extract features and labels
			training_X, training_y = extract_word_feature_and_label(training_list_of_dict, feature_words)
			testing_X, testing_y = extract_word_feature_and_label(testing_list_of_dict, feature_words)
			# DT
			if 'DT' in models:
				dt = DecisionTree(max_depth=maxdep)
				dt.train(training_X, training_y)
				losses[models.index('DT'),i,j] = dt.test(testing_X, testing_y)
			# BDT
			if 'BDT' in models:
				bdt = BaggedDecisionTrees(max_depth=maxdep, n_estimators=ntrees)
				bdt.train(training_X, training_y)
				losses[models.index('BDT'),i,j] = bdt.test(testing_X, testing_y)
			# BODT
			if 'BODT' in models:
				bodt = BoostedDecisionTrees(max_depth=maxdep, n_estimators=ntrees)
				bodt.train(training_X, training_y)
				losses[models.index('BODT'),i,j] = bodt.test(testing_X, testing_y)
			# RF
			if 'RF' in models:
				rf = RandomForest(max_depth=maxdep, n_estimators=ntrees)
				rf.train(training_X, training_y)
				losses[models.index('RF'),i,j] = rf.test(testing_X, testing_y)
			# SVM
			if 'SVM' in models:
				svm = SupportVectorMachine()
				svm.train(training_X, training_y)
				losses[models.index('SVM'),i,j] = svm.test(testing_X, testing_y)
			# Libary functions
			if debug:
				training_y[training_y==0] = -1
				testing_y[testing_y==0] = -1
				skdt = skDecisionTree(max_depth=maxdep, min_samples_split=10)
				skdt.fit(training_X.T, training_y)
				sklosses[0,i,j] = 1-skdt.score(testing_X.T, testing_y)
				print "ZERO-ONE-LOSS-SKDT %.4f" % sklosses[0,i,j]
				skrf = skRandomForest(max_depth=maxdep, n_estimators=ntrees, min_samples_split=10)
				skrf.fit(training_X.T, training_y)
				sklosses[1,i,j] = 1-skrf.score(testing_X.T, testing_y)
				print "ZERO-ONE-LOSS-SKRF %.4f" % sklosses[1,i,j]
		i += 1
	save(losses_file_name, losses)
	save('debug_' + losses_file_name, sklosses)


def analysis_1(models, tssp, num_words, max_depth, n_estimators, debug=False):
	'''
		Analysis across tssp.
	'''
	csv_file_name = 'yelp_data.csv'
	losses_file_name = 'a1_losses.npy'
	cross_validate(csv_file_name, losses_file_name, models, tssp, num_words, max_depth, n_estimators, debug=debug)
	# T-test
	losses = load(losses_file_name)
	# _, pvalues = ttest_ind(losses[models[0]].T, losses[models[1]].T)
	# print "P-values are", pvalues
	# print "Average P is", mean(pvalues)
	# Calculate mean and standard error
	means = mean(losses, axis=2) # #models, #tss
	stds = std(losses, axis=2) # #models, #tss
	sterrs = stds / sqrt(10)
	# Plot
	fig = figure()
	ax = fig.add_subplot(111)
	for i, model, color in [(0,'DT','r'), (1,'BDT','g'), (2,'BODT','k'), (3,'RF','b'), (4,'SVM','c')]:
		ax.errorbar(tssp,means[i],sterrs[i],c=color,marker='o', label=model)
	if debug:
		sklosses = load('debug_' + losses_file_name)
		skmeans = mean(sklosses, axis=2) # #models, #tss
		skstds = std(sklosses, axis=2) # #models, #tss
		sksterrs = skstds / sqrt(10)
		for i, model, color in [(0,'skDT','y'), (1,'skRF','m')]:	
			ax.errorbar(tssp,skmeans[i],sksterrs[i],c=color,marker='o', label=model)
	ax.legend(loc=1)
	ax.set_xlabel('Portion')
	ax.set_ylabel('Loss')
	ax.set_title('Training Set Size v.s. Zero-one-loss')
	show()

def analysis_2(models, tssp, num_words, max_depth, n_estimators, debug=False):
	'''
		Analysis across num_words.
	'''
	csv_file_name = 'yelp_data.csv'
	losses_file_name = 'a2_losses.npy'
	cross_validate(csv_file_name, losses_file_name, models, tssp, num_words, max_depth, n_estimators, debug=debug)
	# T-test
	losses = load(losses_file_name)
	# _, pvalues = ttest_ind(losses[models[0]].T, losses[models[1]].T)
	# print "P-values are", pvalues
	# print "Average P is", mean(pvalues)
	# Calculate mean and standard error
	means = mean(losses, axis=2) # #models, #tss
	stds = std(losses, axis=2) # #models, #tss
	sterrs = stds / sqrt(10)
	# Plot
	fig = figure()
	ax = fig.add_subplot(111)
	for i, model, color in [(0,'DT','r'), (1,'BDT','g'), (2,'BODT','k'), (3,'RF','b'), (4,'SVM','c')]:
		ax.errorbar(num_words,means[i],sterrs[i],c=color,marker='o', label=model)
	if debug:
		sklosses = load('debug_' + losses_file_name)
		skmeans = mean(sklosses, axis=2) # #models, #tss
		skstds = std(sklosses, axis=2) # #models, #tss
		sksterrs = skstds / sqrt(10)
		for i, model, color in [(0,'skDT','y'), (1,'skRF','m')]:	
			ax.errorbar(num_words,skmeans[i],sksterrs[i],c=color,marker='o', label=model)
	ax.legend(loc=1)
	ax.set_xlabel('Number of Features')
	ax.set_ylabel('Loss')
	ax.set_title('Training Set Size v.s. Zero-one-loss')
	show()

def analysis_3(models, tssp, num_words, max_depth, n_estimators, debug=False):
	'''
		Analysis across max_depth.
	'''
	csv_file_name = 'yelp_data.csv'
	losses_file_name = 'a3_losses.npy'
	cross_validate(csv_file_name, losses_file_name, models, tssp, num_words, max_depth, n_estimators, debug=debug)
	# T-test
	losses = load(losses_file_name)
	# _, pvalues = ttest_ind(losses[models[0]].T, losses[models[1]].T)
	# print "P-values are", pvalues
	# print "Average P is", mean(pvalues)
	# Calculate mean and standard error
	means = mean(losses, axis=2) # #models, #tss
	stds = std(losses, axis=2) # #models, #tss
	sterrs = stds / sqrt(10)
	# Plot
	fig = figure()
	ax = fig.add_subplot(111)
	for i, model, color in [(0,'DT','r'), (1,'BDT','g'), (2,'BODT','k'), (3,'RF','b'), (4,'SVM','c')]:
		ax.errorbar(max_depth,means[i],sterrs[i],c=color,marker='o', label=model)
	if debug:
		sklosses = load('debug_' + losses_file_name)
		skmeans = mean(sklosses, axis=2) # #models, #tss
		skstds = std(sklosses, axis=2) # #models, #tss
		sksterrs = skstds / sqrt(10)
		for i, model, color in [(0,'skDT','y'), (1,'skRF','m')]:	
			ax.errorbar(max_depth,skmeans[i],sksterrs[i],c=color,marker='o', label=model)
	ax.legend(loc=1)
	ax.set_xlabel('Max Depth')
	ax.set_ylabel('Loss')
	ax.set_title('Training Set Size v.s. Zero-one-loss')
	show()

def analysis_4(models, tssp, num_words, max_depth, n_estimators, debug=False):
	'''
		Analysis across n_estimators.
	'''
	csv_file_name = 'yelp_data.csv'
	losses_file_name = 'a4_losses.npy'
	cross_validate(csv_file_name, losses_file_name, models, tssp, num_words, max_depth, n_estimators, debug=debug)
	# T-test
	losses = load(losses_file_name)
	# _, pvalues = ttest_ind(losses[models[0]].T, losses[models[1]].T)
	# print "P-values are", pvalues
	# print "Average P is", mean(pvalues)
	# Calculate mean and standard error
	means = mean(losses, axis=2) # #models, #tss
	stds = std(losses, axis=2) # #models, #tss
	sterrs = stds / sqrt(10)
	# Plot
	fig = figure()
	ax = fig.add_subplot(111)
	for i, model, color in [(0,'DT','r'), (1,'BDT','g'), (2,'BODT','k'), (3,'RF','b'), (4,'SVM','c')]:
		ax.errorbar(n_estimators,means[i],sterrs[i],c=color,marker='o', label=model)
	if debug:
		sklosses = load('debug_' + losses_file_name)
		skmeans = mean(sklosses, axis=2) # #models, #tss
		skstds = std(sklosses, axis=2) # #models, #tss
		sksterrs = skstds / sqrt(10)
		for i, model, color in [(0,'skDT','y'), (1,'skRF','m')]:	
			ax.errorbar(n_estimators,skmeans[i],sksterrs[i],c=color,marker='o', label=model)
	ax.legend(loc=1)
	ax.set_xlabel('Number of Trees')
	ax.set_ylabel('Loss')
	ax.set_title('Training Set Size v.s. Zero-one-loss')
	show()