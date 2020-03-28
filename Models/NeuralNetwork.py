#!/usr/bin/env python

##############################################
# Author: Nitay Hason
# Artificial Neural Network class
##############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time
import pickle
import os.path

from sklearn.preprocessing import PolynomialFeatures, normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


class NeuralNetwork:
	def __init__(self, learning_rate = 0.01, training_epochs = 10000, kfolds=10, batch_size = 150, threshold = 0.5, test_size = 0.25, degree = 1, drop_columns = [], dataset_path=None, dataset=None):
		# Hyperparameters
		self.dataset		 = dataset
		self.dataset_path    = dataset_path
		self.learning_rate   = learning_rate
		self.training_epochs = training_epochs
		self.kfolds          = kfolds
		self.batch_size      = batch_size
		self.threshold       = threshold
		self.test_size       = test_size
		self.degree          = degree
		self.drop_columns    = drop_columns
		self.X	             = None
		self.y               = None
		self.X_train         = None
		self.X_test          = None
		self.y_train         = None
		self.y_test          = None
		self.idx_train       = None
		self.idx_test        = None
		self.model 			 = None
		self.model_history	 = None
		self.scaler    		 = None

	def set_dataset(self, dataset_path=None, delimiter=',', skip_header=0, usecols=None, dtype=None):
		if dataset_path is not None:
			self.dataset_path = dataset_path
		if self.dataset_path is not None:
			if dtype is None:
				dtype = np.dtype('Float64')
			self.dataset = np.genfromtxt(dataset_path, delimiter=delimiter, skip_header=skip_header, usecols=usecols, dtype=dtype)

	def build(self, verbose=1):
		use_dataset = self.dataset.copy()
		# Drop unwanted columns
		if len(self.drop_columns)>0:
			use_dataset = np.delete(self.dataset, self.drop_columns, 1)
		use_dataset = np.asfarray(use_dataset,np.dtype('Float64'))

		# Normlize the dataset
		self.scaler= MinMaxScaler().fit(use_dataset[:, :-1])
		dataset_norm = self.scaler.transform(use_dataset[:, :-1])

		# Split into features and labels
		self.X, self.y = dataset_norm, np.transpose([use_dataset[:, -1]]).ravel()

		if self.degree > 1:
			polynomial_features= PolynomialFeatures(degree=self.degree, include_bias=False)
			self.X = polynomial_features.fit_transform(self.X)


	def get_layer(self, layer, input_dim=0):
		ret_layer = []
		if layer[1] == "relu":
			if input_dim>0:
				ret_layer.append(tf.keras.layers.Dense(units=layer[0], input_dim=input_dim))
				ret_layer.append(tf.keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
			else:
				ret_layer.append(tf.keras.layers.Dense(units=layer[0]))
				ret_layer.append(tf.keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
		elif layer[1] == "leakyrelu":
			if input_dim>0:
				ret_layer.append(tf.keras.layers.Dense(units=layer[0], input_dim=input_dim))
				ret_layer.append(tf.keras.layers.LeakyReLU(alpha=0.05))
			else:
				ret_layer.append(tf.keras.layers.Dense(units=layer[0]))
				ret_layer.append(tf.keras.layers.LeakyReLU(alpha=0.05))
		elif layer[1] == "sigmoid":
			ret_layer.append(tf.keras.layers.Dense(units=layer[0], activation='sigmoid'))
		return ret_layer

	def train(self, optimizer=0, layers=[(80,"relu"),(80,"relu"),(80,"leakyrelu"),(1,'sigmoid')], verbose=0):
		# Split the data to train and test
		indices = np.arange(self.y.shape[0])
		self.X_train, self.X_test, self.y_train, self.y_test, self.idx_train, self.idx_test = train_test_split(self.X, self.y, indices, stratify=self.y, test_size=self.test_size, random_state=42)

		input_dim  = self.X_train.shape[1]
		self.model = tf.keras.models.Sequential()

		i = 0
		for layer in layers:
			if i==0:
				i = 1
				ret_layer = self.get_layer(layer, input_dim=input_dim)
			else:
				ret_layer = self.get_layer(layer)

			for add in ret_layer:
					self.model.add(add)


		if optimizer == 0:
			optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		else:
			optimizer = tf.keras.optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

		self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
		callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto')

		kf = KFold(n_splits=self.kfolds, random_state=None, shuffle=False)
		kf.get_n_splits(self.X)

		print("Start training")
		start   = time.time()
		for train_index, test_index in kf.split(self.idx_train):
			X_train_fold, X_test_fold = self.X_train[train_index], self.X_train[test_index]
			y_train_fold, y_test_fold = self.y_train[train_index], self.y_train[test_index]

			self.model_history = self.model.fit(X_train_fold, y_train_fold, epochs=self.training_epochs, batch_size=self.batch_size, validation_data=(X_test_fold,y_test_fold), callbacks=[callback], verbose=verbose)
		# self.model_history = self.model.fit(self.X_train, self.y_train, epochs=self.training_epochs, batch_size=self.batch_size, validation_split=self.test_size, callbacks=[callback], verbose=verbose)
		end = time.time()
		print("\nTraining time:")
		print(end - start)


	def predict(self, verbose=1):
		# predict probabilities for test set
		yhat_probs = self.model.predict(self.X_test, verbose=1)
		return self.predict_check(yhat_probs, verbose=verbose)

	def predict_self(self, X, threshold=None, verbose=1):
		if threshold is None:
			threshold = self.threshold
		X_norm = self.scaler.transform(X)
		if self.degree > 1:
			polynomial_features= PolynomialFeatures(degree=self.degree, include_bias=False)
			X_norm = polynomial_features.fit_transform(X_norm)

		yhat_probs = self.model.predict(X_norm, verbose=0)
		yhat_classes = yhat_probs.copy()
		yhat_classes[yhat_classes>=threshold]=np.float64(1)
		yhat_classes[yhat_classes<threshold]=np.float64(0)
		return [yhat_probs,yhat_classes]

	def predict_check(self, predicted_probs, verbose=1):
		# reduce to 1d array
		yhat_probs = predicted_probs[:, 0]
		yhat_classes = yhat_probs.copy()
		yhat_classes[yhat_classes>=self.threshold]=np.float64(1)
		yhat_classes[yhat_classes<self.threshold]=np.float64(0)

		FPR, TPR, _ = roc_curve(self.y_test, yhat_probs)
		AUC = auc(FPR, TPR)

		# accuracy: (tp + tn) / (p + n)
		accuracy    = accuracy_score(self.y_test, yhat_classes)
		# precision tp / (tp + fp)
		precision   = precision_score(self.y_test, yhat_classes)
		# recall: tp / (tp + fn)
		recall      = recall_score(self.y_test, yhat_classes)
		# f1: 2 tp / (2 tp + fp + fn)
		f1          = f1_score(self.y_test, yhat_classes)
		#loss
		loss        = log_loss(self.y_test, yhat_classes, eps=1e-7)
		# kappa
		kappa       = cohen_kappa_score(self.y_test, yhat_classes)
		# ROC AUC
		auc_score   = roc_auc_score(self.y_test, yhat_probs)
		# confusion matrix
		matrix      = confusion_matrix(self.y_test, yhat_classes)
		cr          = classification_report(self.y_test,yhat_classes,digits=3)

		return_dict = {"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1,"loss":loss,"kappa":kappa,"auc":auc_score,"confusion_matrix":matrix,"classification_report":cr}

		if verbose>0:
			print('Accuracy: %f' % accuracy)
			print('Precision: %f' % precision)
			print('Recall: %f' % recall)
			print('F1 score: %f' % f1)
			print('Loss: %f' % loss)
			print('Cohens kappa: %f' % kappa)
			print('ROC AUC: %f' % auc_score)
			print(matrix)
			print(cr)

		return return_dict

	def plot(self):
		# predict probabilities for test set
		yhat_probs = self.model.predict(self.X_test, verbose=0)

		# reduce to 1d array
		yhat_probs = yhat_probs[:, 0]

		yhat_classes = yhat_probs.copy()
		yhat_classes[yhat_classes>=self.threshold]=np.float64(1)
		yhat_classes[yhat_classes<self.threshold]=np.float64(0)

		FPR, TPR, _ = roc_curve(self.y_test, yhat_probs)
		AUC = auc(FPR, TPR)

		plt.figure("Neural Network ROC")
		plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % AUC)
		plt.plot([0, 1], [0, 1], 'r--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.02])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC Curve')
		plt.legend(loc="lower right")

		# summarize history for accuracy
		plt.figure("Neural Network Accuracy Graph")
		plt.plot(self.model_history.history['acc'])
		plt.plot(self.model_history.history['val_acc'])
		plt.title('Model Accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.show()

	def save_model(self, models_name='model'):
		model_file  = models_name+".h5"
		scaler_file = models_name+"_scaler.sav"
		if self.model_history is not None:
			# save model and architecture to single file
			self.model.save(model_file)
			pickle.dump(self.scaler, open(scaler_file, 'wb'))
			print("Saved model to disk")
			return True
		return False

	def load_model(self, models_name='model'):
		model_file  = models_name+".h5"
		scaler_file = models_name+"_scaler.sav"
		if os.path.isfile(model_file) and os.path.isfile(scaler_file):
			# load model and architecture from single file
			self.model = tf.keras.models.load_model(model_file)
			self.scaler = pickle.load(open(scaler_file, 'rb'))
			print("Loaded model from disk")
			# self.model.summary()
			return True
		return False
