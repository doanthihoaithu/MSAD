########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: utils
# @file : evaluator
#
########################################################################

import os
import pickle
from pathlib import Path
from collections import Counter
from time import perf_counter
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from utils.timeseries_dataset import TimeseriesDataset
from utils.config import *

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import numpy as np
import pandas as pd

from utils.utils import compute_weighted_scores, combine_anomaly_scores, compute_metrics


class Evaluator:
	"""A class with evaluation tools
	"""

	def predict(
		self,
		model,
		fnames,
		data_path,
		batch_size=64,
		deep_model=True,
		device='cuda'
	):
		"""Predict function for all the models

		:param model: the object model whose predictions we want
		:param fnames: the names of the timeseries to be predicted
		:param data_path: the path to the timeseries 
			(please check that path and fnames together make the complete path)
		:param batch_size: the batch size used to make the predictions
		:param deep_model:
		:return df: dataframe with timeseries and predictions per time series
		"""

		# Setup
		all_preds = []
		inf_time = []

		loop = tqdm(
			fnames, 
			total=len(fnames),
			desc="Computing",
			unit="files",
			leave=True
		)

		# Main loop
		for fname in loop:
			# Fetch data for this specific timeseries
			data = TimeseriesDataset(
				data_path=data_path,
				fnames=[fname],
				verbose=False
			)

			if deep_model:
				tic = perf_counter()
				preds = self.predict_timeseries(model, data, batch_size=batch_size, device=device)
			else:
				X_val, y_val = data.__getallsamples__().astype('float32'), data.__getalllabels__()
				tic = perf_counter()
				preds = self.predict_timeseries_non_deep(model, X_val, y_val)

			# Compute metric value
			counter = Counter(preds)
			most_voted = counter.most_common(1)
			toc = perf_counter()
			
			# Save info
			# TODO fix multivariate_detector_names
			all_preds.append(multivariate_detector_names[most_voted[0][0]])
			inf_time.append(toc-tic)
		
		fnames = [x[:-4] for x in fnames]

		return pd.DataFrame(data=zip(all_preds, inf_time), columns=["class", "inf"], index=fnames)

	def predict_with_prob(
			self,
			model,
			fnames,
			data_path,
			batch_size=64,
			deep_model=True,
			is_rocket_model=False,
			device='cuda'
	):
		"""Predict function for all the models

		:param model: the object model whose predictions we want
		:param fnames: the names of the timeseries to be predicted
		:param data_path: the path to the timeseries
			(please check that path and fnames together make the complete path)
		:param batch_size: the batch size used to make the predictions
		:param deep_model:
		:return df: dataframe with timeseries and predictions per time series
		"""

		# Setup
		all_preds = []
		inf_time = []

		loop = tqdm(
			fnames,
			total=len(fnames),
			desc="Computing",
			unit="files",
			leave=True
		)

		# Main loop
		preds_with_probs_dict = dict()
		for fname in loop:
			# Fetch data for this specific timeseries
			data = TimeseriesDataset(
				data_path=data_path,
				fnames=[fname],
				verbose=False
			)

			if deep_model:
				tic = perf_counter()
				preds_with_probs = self.predict_timeseries_with_prob(model, data, batch_size=batch_size, device=device)
			else:
				X_val, y_val = data.__getallsamples__().astype('float32'), data.__getalllabels__()
				tic = perf_counter()
				preds_with_probs = self.predict_timeseries_with_prob_non_deep(model, is_rocket_model, X_val, y_val)

			preds_with_probs_dict[fname] = preds_with_probs
		return preds_with_probs_dict
		# 	# Compute metric value
		# 	counter = Counter(preds)
		# 	most_voted = counter.most_common(1)
		# 	toc = perf_counter()
		#
		# 	# Save info
		# 	# TODO fix multivariate_detector_names
		# 	all_preds.append(multivariate_detector_names[most_voted[0][0]])
		# 	inf_time.append(toc - tic)
		#
		# fnames = [x[:-4] for x in fnames]
		#
		# return pd.DataFrame(data=zip(all_preds, inf_time), columns=["class", "inf"], index=fnames)

	def predict_timeseries(self, model, val_data, batch_size, device='cuda', k=1):
		all_preds = []
		
		# Timeseries to batches
		val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

		for (inputs, labels) in val_loader:
			# Move data to the same device as model
			inputs = inputs.type(torch.float32).to(device)
			labels = labels.type(torch.float32).to(device)

			# Make predictions
			outputs = model(inputs.float())

			# Compute topk acc
			preds = outputs.argmax(dim=1)
			all_preds.extend(preds.tolist())

		return all_preds

	def predict_timeseries_with_prob(self, model, val_data, batch_size, device='cuda', k=1):
		all_preds = []

		# Timeseries to batches
		val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
		tensor_softmax = nn.Softmax(dim=1)

		for (inputs, labels) in val_loader:
			# Move data to the same device as model
			inputs = inputs.type(torch.float32).to(device)
			labels = labels.type(torch.float32).to(device)

			# Make predictions
			outputs = model(inputs.float())

			# Compute topk acc
			preds = tensor_softmax(outputs)
			preds =	preds.cpu().detach().numpy()
			all_preds.extend(preds.tolist())
		all_preds = np.array(all_preds)
		assert all_preds.ndim == 2
		return all_preds


	def predict_timeseries_non_deep(self, model, X_val, y_val):
		all_preds = []
		all_acc = []
		
		# Make predictions
		preds = model.predict(X_val)

		# preds = outputs.argmax(dim=1)
		# acc = (y_val == preds).sum() / y_val.shape[0]

		# all_acc.append(acc)
		all_preds.extend(preds.tolist())
		all_preds = np.array(all_preds)

		return all_preds

	def predict_timeseries_with_prob_non_deep(self, model, is_rocket, X_val, y_val):
		all_preds = []
		all_acc = []

		if is_rocket:
			# Make predictions
			preds = model.decision_function(X_val)
		else:
			# Make predictions
			preds = model.predict_proba(X_val)

		# preds = outputs.argmax(dim=1)
		# acc = (y_val == preds).sum() / y_val.shape[0]

		# all_acc.append(acc)
		all_preds.extend(preds.tolist())
		all_preds = np.array(all_preds)

		return all_preds


def save_classifier(model, path, fname=None):
	# Set up
	timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
	fname = f"model_{timestamp}" if fname is None else fname

	# Create saving dir if we need to
	filename = Path(os.path.join(path, f"{fname}.pkl"))
	filename.parent.mkdir(parents=True, exist_ok=True)

	# Save
	with open(filename, 'wb') as output:
		pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

	return str(filename)


def load_classifier(path):
	"""Loads a classifier/model that is a pickle (.pkl) object.
	If the path is only the path to the directory of a given class
	of models, then the youngest model of that class is retrieved.

	:param path: path to the specific classifier to load,
		or path to a class of classifiers (e.g. rocket)
	:return output: the loaded classifier
	"""

	# If model is not given, load the latest
	if os.path.isdir(path):
		models = [x for x in os.listdir(path) if '.pkl' in x]
		models.sort(key=lambda date: datetime.strptime(date, 'model_%d%m%Y_%H%M%S.pkl'))
		path = os.path.join(path, models[-1])
	elif '.pkl' not in path:
		raise ValueError(f"Can't load this type of file {path}. Only '.pkl' files please")

	filename = Path(path)
	with open(f'{filename}', 'rb') as input:
		output = pickle.load(input)
	
	return output

'''
	def predict_non_deep(self, model, X_val, y_val):
		all_preds = []
		all_acc = []
		
		# Make predictions
		preds = model.predict(X_val)

		# preds = outputs.argmax(dim=1)
		# acc = (y_val == preds).sum() / y_val.shape[0]

		# all_acc.append(acc)
		all_preds.extend(preds.tolist())

		return all_preds
'''