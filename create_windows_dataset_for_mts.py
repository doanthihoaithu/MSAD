########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : create_windows_dataset
#
########################################################################


import sys
import os

import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

from utils.data_loader import DataLoader
from utils.metrics_loader import MetricsLoader
from utils.scores_loader import ScoresLoader
from utils.config import *


def create_tmp_dataset(
	name,
	save_dir,
	data_path,
	metric_path,
	window_sizes,
	metric,
	mode
):
	"""Generates a new dataset from the given dataset. The time series
	in the generated dataset have been divided in windows.

	:param name: the name of the experiment
	:param save_dir: directory in which to save the new dataset
	:param data_path: path to dataset to be divided
	:param window_size: the size of the window timeseries will be split to
	:param metric: the specific metric to read
	"""

	# Load datasets
	dataloader = DataLoader(data_path)
	datasets = dataloader.get_dataset_names()
	x, y, fnames = dataloader.load(datasets)

	# Load metrics
	metricsloader = MetricsLoader(metric_path)
	metrics_data_dict = {}
	if metric == 'all':
		metrics_data_dict = metricsloader.read_multiple_metrics(supported_metrics_for_labeling)
	else:
		metrics_data_dict[metric.upper()] = metricsloader.read(metric.upper())

	for window_size in (progBar := tqdm(window_sizes, total=len(window_sizes), desc='Creating windowed dataset ...')):
		progBar.set_postfix_str(f'window_size={window_size}')
		# Form new dataset's name
		name = '{}_{}'.format(name, window_size)

		# Delete any data not in metrics (some timeseries metric scores were not computed)
		idx_to_delete = set()
		for metrics_data in metrics_data_dict.values():
			idx_to_delete.update([i for i, x in enumerate(fnames) if x not in metrics_data.index])
		idx_to_delete = list(idx_to_delete)

		# Delete any time series shorter than requested window
		idx_to_delete_short = [i for i, ts in enumerate(x) if ts.shape[0] < window_size]
		if len(idx_to_delete_short) > 0:
			print(">>> Window size: {} too big for some timeseries. Deleting {} timeseries"
					.format(window_size, len(idx_to_delete_short)))
			idx_to_delete.extend(idx_to_delete_short)

		if len(idx_to_delete) > 0:
			for idx in sorted(idx_to_delete, reverse=True):
				del x[idx]
				del y[idx]
				del fnames[idx]

		# TODO can be optimized by using type of numpy array [num_metric, num_windows, MTS] but need to change code
		metrics_data_dict = {key:metrics_data.loc[fnames] for key, metrics_data in metrics_data_dict.items()}
		assert(
			list(metrics_data_dict[next(iter(metrics_data_dict))].index) == fnames
		)

		# Keep only the metrics of the detectors (remove oracles)
		# metrics_data = metrics_data[univariate_detector_names] if mode == 'univariate' else metrics_data[multivariate_detector_names]
		metrics_data_dict = {key:metrics_data.loc[univariate_detector_names] if mode == 'univariate' else metrics_data[multivariate_detector_names]
							 for key, metrics_data in metrics_data_dict.items()}

		# Split timeseries and compute labels
		# ts_list, labels = split_and_compute_labels(x, metrics_data_dict['AUC_PR'], window_size)
		ts_list, label_by_metric_dict = split_and_compute_multiple_labels(x, metrics_data_dict, window_size)
		# Uncomment to check the results
		# fig, axs = plt.subplots(2, 1, sharex=True)
		# x_new = np.concatenate(ts_list[3])
		# print(np.mean(x_new))
		# print(np.std(x_new))
		# axs[0].plot(x_new)
		# axs[1].plot(x[3])
		# plt.show()

		# Create subfolder for each dataset
		for dataset in datasets:
			Path(os.path.join(save_dir, name, dataset)).mkdir(parents=True, exist_ok=True)

		# Save new dataset
		for ts_index, ts, fname in tqdm(zip(range(len(ts_list)), ts_list, fnames),
										total=len(ts_list),
										desc='Save dataset',
										position=0,
										leave=True):
			fname_split = fname.split('/')
			dataset_name = fname_split[-2]
			ts_name = fname_split[-1]
			new_names = [ts_name + '.{}'.format(i) for i in range(len(ts))]

			# data = np.concatenate((label[:, np.newaxis], ts), axis=1)
			# col_names = ['label']

			multiple_labels = np.stack([label_list[ts_index] for label_list in label_by_metric_dict.values()], axis=1)
			data = np.concatenate((multiple_labels, ts), axis=1)
			col_names = [f'label_by_{f}' for f in label_by_metric_dict.keys()]

			num_features = (data.shape[1] - len(col_names)) // window_size
			for i in range(len(col_names), data.shape[1]):
				col_names.append(f'val_{(i-1)//num_features}_dim_{(i-1)%num_features}')

			df = pd.DataFrame(data, index=new_names, columns=col_names)
			df.to_csv(os.path.join(save_dir, name, dataset_name, ts_name + '.csv'))


def split_and_compute_labels(x, metrics_data, window_size):
	'''Splits the timeseries, computes the labels and returns 
	the segmented timeseries and the new labels.

	:param x: list of the timeseries to be segmented (as np arrays)
	:param metrics_data: df with the scores of all the detectors for every time series
	:param window_size: the size of the windows that will be created
	:return ts_list: list of n 2D arrays (n is number of time series in x)
	:return labels: labels for every created window
	'''
	ts_list = []
	labels = []

	assert(
		len(x) == metrics_data.shape[0]
	), "Lengths and shapes do not match. Please check"

	for ts, metric_label in (pbar:= tqdm(zip(x, metrics_data.idxmax(axis=1)), total=len(x), desc="Split dataset to windows...",
								 position=0, leave=True)):

		if ts.ndim > 1:
			for i in range(ts.shape[1]):
				ts[:, i] = z_normalization(ts[:, i], decimals=7)
		else:
			# Z-normalization (windows with a single value go to 0)
			ts = z_normalization(ts, decimals=7)

		# Split time series into windows
		ts_split = split_ts(ts, window_size)
		
		# Save everything to lists
		ts_list.append(ts_split)
		labels.append(np.ones(len(ts_split)) * multivariate_detector_names.index(metric_label))

	assert(
		len(x) == len(ts_list) == len(labels)
	), "Timeseries split and labels computation error, lengths do not match"
			
	return ts_list, labels


def split_and_compute_multiple_labels(x, metrics_data_dict, window_size)->tuple[list, dict]:
	'''Splits the timeseries, computes the labels and returns
	the segmented timeseries and the new labels.

	:param x: list of the timeseries to be segmented (as np arrays)
	:param metrics_data: df with the scores of all the detectors for every time series
	:param window_size: the size of the windows that will be created
	:return ts_list: list of n 2D arrays (n is number of time series in x)
	:return labels: labels for every created window
	'''

	ts_list = []
	labels = []
	labels_dict = dict({metric_key: [] for metric_key in metrics_data_dict})

	assert (
			len(x) == metrics_data_dict[next(iter(metrics_data_dict.keys()))].shape[0]
	), "Lengths and shapes do not match. Please check"

	for ts_index, ts in (pbar:= tqdm(enumerate(x), total=len(x), desc=f'Split dataset to windows of length {window_size}...',
							 position=0, leave=True)):

		if ts.ndim > 1:
			for i in range(ts.shape[1]):
				ts[:, i] = z_normalization(ts[:, i], decimals=7)
		else:
			# Z-normalization (windows with a single value go to 0)
			ts = z_normalization(ts, decimals=7)

		# Split time series into windows
		ts_split = split_ts(ts, window_size)

		# Save everything to lists
		ts_list.append(ts_split)
		# labels.append(np.ones(len(ts_split)) * multivariate_detector_names.index(metric_label))
		for metric_key, metrics_data in metrics_data_dict.items():
			labels = np.ones(len(ts_split)) * multivariate_detector_names.index(metrics_data.idxmax(axis=1).iloc[ts_index])
			labels_dict[metric_key].append(labels)

	# labels_dict = {key: np.ones(len(ts_split)) * multivariate_detector_names.index(metrics_data.idxmax(axis=1))
	# 			   for key, metrics_data in metrics_data_dict.items()}


	assert (
			len(x) == len(ts_list) == len(labels_dict[next(iter(labels_dict))])
	), "Timeseries split and labels computation error, lengths do not match"

	return ts_list, labels_dict


def z_normalization(ts, decimals=5):
	# Z-normalization (all windows with the same value go to 0)
	if len(set(ts)) == 1:
		ts = ts - np.mean(ts)
	else:
		ts = (ts - np.mean(ts)) / np.std(ts)
	ts = np.around(ts, decimals=decimals)

	# Test normalization
	assert(
		np.around(np.mean(ts), decimals=3) == 0 and np.around(np.std(ts) - 1, decimals=3) == 0
	), "After normalization it should: mean == 0 and std == 1"

	return ts

def split_ts(data, window_size):
	'''Split a timeserie into windows according to window_size.
	If the timeserie can not be divided exactly by the window_size
	then the first window will overlap the second.

	:param data: the timeserie to be segmented
	:param window_size: the size of the windows
	:return data_split: an 2D array of the segmented time series
	'''

	# Compute the modulo
	modulo = data.shape[0] % window_size

	# Compute the number of windows
	k = data[modulo:].shape[0] / window_size
	assert(math.ceil(k) == k)

	data_split = np.split(data[modulo:], k)
	if modulo != 0:
		if data.ndim == 1 or data.shape[1] == 1:
			data_split.insert(0, list(data[:window_size]))
		else:
			data_split.insert(0, data[:window_size, :])
			data_split = np.asarray(data_split)

			if data_split.ndim == 3:
				k, window_size, n_features = data_split.shape
				data_split = data_split.reshape(k, window_size * n_features)
			else:
				data_split = data_split
	else:
		data_split = np.asarray(data_split)
		if data_split.ndim == 3:
			k, window_size, n_features = data_split.shape
			data_split = data_split.reshape(k, window_size * n_features)
		else:
			data_split = data_split

	return data_split


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
	if cfg.create_windows_dataset.window_size == "all":
		window_sizes = cfg.supported_window_sizes

		create_tmp_dataset(
			name=cfg.create_windows_dataset.dataset_name,
			save_dir=cfg.create_windows_dataset.save_dir,
			data_path=cfg.create_windows_dataset.path,
			metric_path=cfg.create_windows_dataset.metric_path,
			window_sizes=window_sizes,
			metric=cfg.create_windows_dataset.metric_for_labeling_windows,
			mode='multivariate'
		)
	else:
		create_tmp_dataset(
			name=cfg.create_windows_dataset.dataset_name,
			save_dir=cfg.create_windows_dataset.save_dir,
			data_path=cfg.create_windows_dataset.path,
			metric_path=cfg.create_windows_dataset.metric_path,
			window_sizes=[cfg.create_windows_dataset.window_size],
			metric=cfg.create_windows_dataset.metric_for_labeling_windows,
			mode='multivariate'
		)

if __name__ == "__main__":
	# parser = argparse.ArgumentParser(
	# 	prog='Create temporary/experiment-specific dataset',
	# 	description='This function creates a dataset of the size you want.  The data that will be used are set into the config file',
	# 	epilog='Be careful where you save the generated dataset'
	# )
	#
	# parser.add_argument('-n', '--name', type=str, help='path to save the dataset', default="TSB")
	# parser.add_argument('-s', '--save_dir', type=str, help='path to save the dataset', required=True)
	# parser.add_argument('-p', '--path', type=str, help='path of the dataset to divide', required=True)
	# parser.add_argument('-mp', '--metric_path', type=str, help='path to the metrics of the dataset given', default=TSB_metrics_path)
	# parser.add_argument('-w', '--window_size', type=str, help='window size to segment the timeseries to', required=True)
	# parser.add_argument('-m', '--metric', type=str, help='metric to use to produce the labels', default='AUC_PR')
	#
	# args = parser.parse_args()

	# --name = settings_one
	# --save_dir = data/mts/settings_one
	# --path = data/mts/settings_one/data/
	# --metric_path = data/mts/settings_one/metrics/
	# --window_size = 32
	# --metric = AUC_ROC
	main()


