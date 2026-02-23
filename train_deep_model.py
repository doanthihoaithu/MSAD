########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : train_deep_model
#
########################################################################

import argparse
import os
import re

import hydra
from datetime import datetime

import numpy as np
import pandas as pd
from omegaconf import DictConfig, open_dict

from utils.train_deep_model_utils import ModelExecutioner, json_file

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.timeseries_dataset import create_splits, TimeseriesDataset
from utils.config import *
from eval_deep_model import eval_deep_model


def train_deep_model(
	data_path,
	num_dimensions,
	metric_for_optimization,
	model_name,
	split_per,
	seed,
	read_from_file,
	batch_size,
	model_parameters_file,
	epochs,
	eval_model=False,
	path_model_save=None,
	save_done_training=None,
	path_prediction_save=None,
	path_save_runs= None
):
	os.makedirs(path_model_save, exist_ok=True)
	os.makedirs(path_prediction_save, exist_ok=True)
	os.makedirs(path_save_runs, exist_ok=True)
	os.makedirs(save_done_training, exist_ok=True)

	# Set up
	window_size = int(re.search(r'(\d+)$', str(data_path)).group())
	# data_path example: 'data/mts/settings_one/settings_one_32'
	working_dataset = '_'.join(str(data_path).split('/')[-1].split('_')[:-1])
	device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
	# save_runs = f'results/runs/{working_dataset}'
	# save_weights = f'results/weights/{working_dataset}'
	save_runs = path_save_runs
	save_weights = path_model_save
	inf_time = True 		# compute inference time per timeseries

	# Load the splits
	train_set, val_set, test_set = create_splits(
		data_path,
		split_per=split_per,
		seed=seed,
		read_from_file=read_from_file,
	)
	# Uncomment for testing
	if epochs == 1:
		train_set, val_set, test_set = train_set[:50], val_set[:10], test_set[:10]

	# Load the data
	print('----------------------------------------------------------------')
	print(f'Current metric for optimization: {metric_for_optimization}, => Loading data with label_by={metric_for_optimization}')
	training_data = TimeseriesDataset(data_path, num_dimensions=num_dimensions, label_by=metric_for_optimization, fnames=train_set)
	val_data = TimeseriesDataset(data_path, num_dimensions=num_dimensions,  label_by=metric_for_optimization, fnames=val_set)
	test_data = TimeseriesDataset(data_path, num_dimensions=num_dimensions, label_by=metric_for_optimization, fnames=test_set)
	
	# Create the data loaders
	training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
	validation_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

	# Compute class weights to give them to the loss function
	class_weights = training_data.get_weights_subset(device)

	# Read models parameters
	model_parameters = json_file(model_parameters_file)
	model_parameters
	
	# Change input size according to input
	if 'original_length' in model_parameters:
		model_parameters['original_length'] = window_size
	if 'timeseries_size' in model_parameters:
		model_parameters['timeseries_size'] = window_size
	if 'original_dim' in model_parameters:
		model_parameters['original_dim'] = num_dimensions
	if 'in_channels' in model_parameters:
		model_parameters['in_channels'] = num_dimensions
	
	# Create the model, load it on GPU and print it
	model = deep_models[model_name.lower()](**model_parameters).to(device)
	classifier_name = f"{model_parameters_file.split('/')[-1].replace('.json', '')}_{window_size}"
	if read_from_file is not None and "unsupervised" in read_from_file:
		classifier_name += f"_{read_from_file.split('/')[-1].replace('unsupervised_', '')[:-len('.csv')]}"
	
	# Create the executioner object
	model_execute = ModelExecutioner(
		model=model,
		model_name=classifier_name,
		device=device,
		criterion=nn.CrossEntropyLoss(weight=class_weights).to(device),
		runs_dir=save_runs,
		weights_dir=save_weights,
		learning_rate=0.00001
	)

	# Check device of torch
	model_execute.torch_devices_info()

	# Run training procedure
	model, results = model_execute.train(
		n_epochs=epochs, 
		training_loader=training_loader, 
		validation_loader=validation_loader, 
		verbose=True,
	)

	# Save training stats
	timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
	df = pd.DataFrame.from_dict(results, columns=["training_stats"], orient="index")
	df.to_csv(os.path.join(save_done_training, f"{classifier_name}_{timestamp}.csv"))

	# Evaluate on test set or val set
	if eval_model:
		if read_from_file is not None and "unsupervised" in read_from_file:
			os.path.join(path_prediction_save, "unsupervised")
		eval_set = test_set if len(test_set) > 0 else val_set
		# TODO fix hardcode
		eval_deep_model(
			data_path=data_path,
			num_dimensions=num_dimensions,
			metric_for_optimization=metric_for_optimization,
			fnames=eval_set,
			model_name=model_name,
			model=model,
			path_save=path_prediction_save,
		)

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
	train_deep_model_config = cfg.model_selection.deep_model_config
	current_metric_for_optimization = cfg.model_selection.mts_current_metric_for_optimization
	if train_deep_model_config.model_name == 'all':
		for model_name in ['resnet_default','convnet_default', 'inception_time_default' ]:
			with open_dict(cfg):
				cfg.model_selection.deep_model_config.model_name = model_name
			# model_parameters_file = train_deep_model_config.model_parameters_file.replace('all.json',
			# 																			  f'{model_name}.json')
			model_parameters_file = train_deep_model_config.model_parameters_file
			# cfg.update({'model_selection': {'deep_model_config': {'model_parameters_file': model_parameters_file}}})
			if cfg.run_all_windows == False:
				train_deep_model(
					data_path=train_deep_model_config.data_path,
					num_dimensions=train_deep_model_config.num_dimensions,
					metric_for_optimization=current_metric_for_optimization,
					split_per=train_deep_model_config.split_per,
					seed=train_deep_model_config.seed,
					read_from_file=train_deep_model_config.read_from_file,
					model_name=model_name,
					model_parameters_file=model_parameters_file,
					batch_size=train_deep_model_config.batch_size,
					epochs=train_deep_model_config.epochs,
					eval_model=train_deep_model_config.eval_model,
					path_model_save=train_deep_model_config.path_model_save,
					save_done_training=train_deep_model_config.save_done_training,
					path_prediction_save=train_deep_model_config.path_prediction_save,
					path_save_runs = train_deep_model_config.path_save_runs,
				)
			else:
				window_sizes = cfg.supported_window_sizes
				for window_size in window_sizes:
					# model_parameters_file = train_deep_model_config.model_parameters_file

					print(f'\n\n\nTraining model: {model_name} for window size: {window_size}\n\n\n')
					data_path = train_deep_model_config.data_path_template.format(current_window_size=window_size)
					split_file = train_deep_model_config.read_from_file_template.format(current_window_size=window_size)
					train_deep_model(
						data_path=data_path,
						num_dimensions=train_deep_model_config.num_dimensions,
						metric_for_optimization=current_metric_for_optimization,
						split_per=train_deep_model_config.split_per,
						seed=train_deep_model_config.seed,
						read_from_file=split_file,
						model_name=model_name,
						model_parameters_file=model_parameters_file,
						batch_size=train_deep_model_config.batch_size,
						epochs=train_deep_model_config.epochs,
						eval_model=train_deep_model_config.eval_model,
						path_model_save=train_deep_model_config.path_model_save,
						save_done_training=train_deep_model_config.save_done_training,
						path_prediction_save=train_deep_model_config.path_prediction_save,
						path_save_runs=train_deep_model_config.path_save_runs,
					)

	else:
		train_deep_model(
			data_path=train_deep_model_config.data_path,
			num_dimensions=train_deep_model_config.num_dimensions,
			metric_for_optimization=current_metric_for_optimization,
			split_per=train_deep_model_config.split_per,
			seed=train_deep_model_config.seed,
			read_from_file=train_deep_model_config.read_from_file,
			model_name=train_deep_model_config.model_name,
			model_parameters_file=train_deep_model_config.model_parameters_file,
			batch_size=train_deep_model_config.batch_size,
			epochs=train_deep_model_config.epochs,
			eval_model=train_deep_model_config.eval_model,
			path_model_save=train_deep_model_config.path_model_save,
			save_done_training=train_deep_model_config.save_done_training,
			path_prediction_save=train_deep_model_config.path_prediction_save,
			path_save_runs = train_deep_model_config.path_save_runs,
		)

if __name__ == "__main__":
	# parser = argparse.ArgumentParser(
	# 	prog='run_experiment',
	# 	description='This function is made so that we can easily run configurable experiments'
	# )
	#
	# parser.add_argument('-p', '--path', type=str, help='path to the dataset to use', required=True)
	# parser.add_argument('-s', '--split', type=float, help='split percentage for train and val sets', default=0.7)
	# parser.add_argument('-se', '--seed', type=int, default=None, help='Seed for train/val split')
	# parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)
	# parser.add_argument('-m', '--model', type=str, help='model to run', required=True)
	# parser.add_argument('-pa', '--params', type=str, help="a json file with the model's parameters", required=True)
	# parser.add_argument('-b', '--batch', type=int, help='batch size', default=64)
	# parser.add_argument('-ep', '--epochs', type=int, help='number of epochs', default=10)
	# parser.add_argument('-e', '--eval-true', action="store_true", help='whether to evaluate the model on test data after training')
	#
	# args = parser.parse_args()
	# --path=data/mts/settings_two/settings_two_32
	# --split=0.5
	# --file=data/mts/settings_two/settings_two_32/supervised_splits/train_test_split.csv
	# --model=resnet
	# --params=models/configuration/resnet_default.json
	# --batch=256
	# --epochs=100
	# --eval-true

	# --path=data/mts/settings_two/settings_two_32
	# --split=0.5
	# --file=data/mts/settings_two/settings_two_32/supervised_splits/train_test_split.csv
	# --model=convnet
	# --params=models/configuration/convnet_default.json
	# --batch=256
	# --epochs=100
	# --eval-true
	main()

