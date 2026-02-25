########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : eval_deep_model
#
########################################################################

import argparse
import itertools
import re
import os
from collections import Counter

import hydra
import torch
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm

from train_feature_based import classifiers
from utils.config import *
from utils.scores_loader import ScoresLoader
from utils.train_deep_model_utils import json_file
from utils.timeseries_dataset import read_files, create_splits
from utils.evaluator import Evaluator, load_classifier
from utils.utils import get_project_root, compute_weighted_scores, compute_metrics, combine_anomaly_scores


def eval_combine_multiple_detectors(combine_detector_evaluation_config,
									# data_path, #Path to mts/settings_five/data
									# model_name,
									# model_path=None,
									# model_parameters_file=None,
									# path_save=None,
									# fnames=None,
									# read_from_file=None,
									# model=None
									):
	"""Evaluate a deep learning model on time series data and predict the time series.

    :param data_path: Path to the time series data.
    :param model_name: Name of the model to be evaluated.
    :param model_path: Path to the pretrained model weights.
    :param model_parameters_file: Path to the JSON file containing model parameters.
    :param path_save: Path to save the evaluation results.
    :param fnames: List of file names (time series) to evaluate.
    :param read_from_file: Path to the file containing split information.
    :param model: Preloaded model instance.

	Returns:
    DataFrame: A DataFrame containing the predicted time series.
	"""
	running_mode = combine_detector_evaluation_config.running_mode
	project_root = get_project_root()
	data_path = combine_detector_evaluation_config.mts_current_data_path
	mts_running_dataset = combine_detector_evaluation_config.mts_running_dataset
	print(f"Current dataset: {mts_running_dataset}, Running mode: {running_mode}")
	scores_path = combine_detector_evaluation_config.mts_current_scores_path
	scoreloader = ScoresLoader(scores_path)
	print("Init ScoresLoader with path:", scores_path)
	read_from_file = combine_detector_evaluation_config.train_test_split_file
	path_save = combine_detector_evaluation_config.path_prediction_save #Path save prediction results

	# Load the splits
	if read_from_file is not None:
		print("Extracting training and testing files in train_test_split at:", read_from_file)
		_, val_set, test_set = create_splits(
			data_path,
			read_from_file=read_from_file,
		)
		fnames = test_set if len(test_set) > 0 else val_set
		fnames_for_loading_scores = [f[:-4] for f in fnames]  # Remove .csv extension

		print("Number of time series to evaluate:", len(fnames_for_loading_scores))

		print("Loading scores for the evaluated time series...")
		# scores, idx_failed = scoreloader.load(fnames_for_loading_scores)
		scores, contribution_scores, idx_failed = scoreloader.load_multivariate_score_per_var(fnames_for_loading_scores)
		print(f'Scores for {len(fnames_for_loading_scores)} evaluated MTS loaded.')
		if len(idx_failed) > 0:
			raise ValueError(f"Failed to load scores for the following time series: {idx_failed}")
		print(f'Loading ground truth labels for {len(fnames_for_loading_scores)} evaluated time series...')
		univarate_labels_dict, univariate_label_idx_failed = scoreloader.load_univariate_labels(fnames_for_loading_scores)
		multivariate_labels_dict, multivariate_label_idx_failed = scoreloader.load_multivariate_labels(fnames_for_loading_scores)
		print(f'Ground truth labels for {len(fnames_for_loading_scores)} evaluated time series loaded.')

		if running_mode == 'all':
			supported_model_families = combine_detector_evaluation_config.supported_model_families
			model_names = []
			for running_model_family in supported_model_families:
				if running_model_family == 'deep':
					model_names.extend(['resnet_default', 'convnet_default', 'inception_time_default'])
				elif running_model_family == 'transformer':
					model_names.extend(['sit_conv_patch', 'sit_linear_patch', 'sit_stem_original', 'sit_stem_relu'])
				elif running_model_family == 'feature_based':
					model_names.extend(list(classifiers.keys()))
				else:
					# model_names.extend(['rocket'])
					print(f"Skip rocket...")

		else:
			assert running_mode == 'single', "Invalid running mode. Choose either 'all' or 'single'."
			running_model_family = combine_detector_evaluation_config.running_model_family
			assert running_model_family in ['deep','transformer', 'feature_based', 'rocket'], "Invalid model family. Choose either 'deep' or 'transformer'."
			model_names = []
			if running_model_family == 'deep':
				model_names = ['resnet_default', 'convnet_default', 'inception_time_default']
			elif running_model_family == 'transformer':
				# model_names = ['sit_conv_patch', 'sit_linear_patch', 'sit_stem_original', 'sit_stem_relu']
				model_names = ['sit_conv_patch', 'sit_linear_patch', 'sit_stem_original']
			elif running_model_family == 'feature_based':
				model_names = list(classifiers.keys())
				# model_names = ['qda']
			else:
				model_names = ['rocket']

		print("Model names to evaluate:", model_names)

		supported_window_sizes = combine_detector_evaluation_config.supported_window_sizes
		print("Supported window sizes:", supported_window_sizes)
		for window_size in supported_window_sizes:
			windowed_data_folder_name = f'{mts_running_dataset}_{window_size}'
			windowed_data_path = os.path.join(os.path.dirname(data_path), windowed_data_folder_name)

			extracted_features_data_path = os.path.join(windowed_data_path, f'TSFRESH_{mts_running_dataset}_{window_size}.csv')

			# model_save_dir = train_deep_model_config.path_model_save
			# detected_window_size = int(re.search(r'(\d+)$', train_deep_model_config.data_path).group())
			for model_name in model_names:
				# model_short_name = train_deep_model_config.model_name
				# if model_name in ['resnet_default', 'convnet_default', 'inception_time_default']:
				# 	model_full_name = model_name + f"_default_{window_size}"
				# else:
				model_full_name = model_name + f"_{window_size}"
				model_save_dir = combine_detector_evaluation_config.path_model_save
				model_save_dir = os.path.join(model_save_dir, model_full_name)

				all_subdirs = [os.path.join(project_root, model_save_dir, d) for d in os.listdir(model_save_dir) if
							   os.path.isfile(os.path.join(model_save_dir, d))]
				latest_file = max(all_subdirs, key=os.path.getmtime)
				model_path = os.path.join(model_save_dir, latest_file)

				batch_size = 128
				if model_name in deep_models.keys():
					model_parameters_file_template = combine_detector_evaluation_config.model_parameters_file_template
					model_parameters_file = model_parameters_file_template.format(model_name=model_name)

					# window_size = int(re.search(r'\d+', str(data_path)).group())

					# assert (
					# 		(model is not None) or \
					# 		(model_path is not None and model_parameters_file is not None)
					# ), "You should provide the model or the path to the model, not both"
					#
					# assert (
					# 	not (fnames is not None and read_from_file is not None)
					# ), "You should provide either the fnames or the path to the specific splits, not both"

					# Load the model only if not provided
					# if model == None:
						# Read models parameters
					model_parameters = json_file(model_parameters_file)

					# Change input size according to input
					if 'original_length' in model_parameters:
						model_parameters['original_length'] = window_size
					if 'timeseries_size' in model_parameters:
						model_parameters['timeseries_size'] = window_size
					if 'in_channels' in model_parameters:
						model_parameters['in_channels'] = combine_detector_evaluation_config.num_dimensions
					if 'original_dim' in model_parameters:
						model_parameters['original_dim'] = combine_detector_evaluation_config.num_dimensions

					# Load model
					model = deep_models[model_name](**model_parameters)

					# Check if model_path is specific file or dir
					if os.path.isdir(model_path):
						# Check the number of files in the directory
						files = os.listdir(model_path)
						if len(files) == 1:
							# Load the single file from the directory
							model_path = os.path.join(model_path, files[0])
						else:
							raise ValueError(
								"Multiple files found in the 'model_path' directory. Please provide a single file or specify the file directly.")

					if torch.cuda.is_available():
						model.load_state_dict(torch.load(model_path))
						model.eval()
						model.to('cuda')
					elif torch.backends.mps.is_available():
						model.load_state_dict(torch.load(model_path))
						model.eval()
						model.to('mps')
					else:
						model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
						model.eval()
				elif model_name in classifiers.keys() or model_name == 'rocket':
					model = load_classifier(model_path)
				else:
					raise ValueError(f"Model {model_name} not supported for combining detectors.")

				# Uncomment for testing
				# fnames = fnames[:10]

				# Specify classifier name for saving results
				# if model_path is not None:
				# 	if "sit_conv" in model_path:
				# 		model_name = "sit_conv"
				# 	elif "sit_linear" in model_path:
				# 		model_name = "sit_linear"
				# 	elif "sit_stem_relu" in model_path:
				# 		model_name = "sit_stem_relu"
				# 	elif "sit_stem" in model_path:
				# 		model_name = "sit_stem"
				# classifier_name = f"{model_name}_{window_size}"
				# if read_from_file is not None and "unsupervised" in read_from_file:
				# 	classifier_name += f"_{read_from_file.split('/')[-1].replace('unsupervised_', '')[:-len('.csv')]}"
				# elif "unsupervised" in path_save:
				# 	extra = model_path.split('/')[-2].replace(classifier_name, "")
				# 	classifier_name += extra

				# Evaluate model
				evaluator = Evaluator()
				# results = evaluator.predict(
				# 	model=model,
				# 	fnames=fnames,
				# 	data_path=data_path,
				# 	batch_size=batch_size,
				# 	deep_model=True,
				# 	device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
				# )
				window_pred_probabilities = evaluator.predict_with_prob(
					model=model,
					fnames=fnames,
					num_dimensions=combine_detector_evaluation_config.num_dimensions,
					metric_for_optimization=combine_detector_evaluation_config.mts_current_metric_for_optimization,
					data_path=windowed_data_path if (model_name in deep_models.keys() or model_name == 'rocket') else extracted_features_data_path,
					batch_size=batch_size,
					deep_model=True if model_name in deep_models.keys() else False,
					is_rocket_model= True if model_name == 'rocket' else False,
					device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
					n_detectors=len(scoreloader.get_detector_names())
				)
				# print(window_pred_probabilities.keys())
				top_k_list = combine_detector_evaluation_config.top_k_list
				combine_strategy_list = combine_detector_evaluation_config.strategy_list
				curr_top_combinations = list(itertools.product(top_k_list, combine_strategy_list))

				# top_models_combinations = {
				# 	'convnet': [(2, 'vote'), (2, 'average')],
				# 	'resnet': [(4, 'vote'), (5, 'average')],
				# 	'sit': [(7, 'vote'), (8, 'average')],
				# 	'knn': [(3, 'vote'), (8, 'average')],
				# }
				# curr_top_combinations = top_models_combinations['convnet']
				for k, combination_method in (pbar := tqdm(curr_top_combinations,
														   desc=f'Combining detectors for {model_name} with window size {window_size}',
														   total=len(curr_top_combinations))):
					pbar.set_postfix({'k': k, 'method': combination_method})

					weights = compute_weighted_scores(window_pred_probabilities.values(), combination_method, k)
					weighted_scores, weighted_contribution_scores = combine_anomaly_scores(scores, contribution_scores, weights, plot=False)
					metric_values_dict = {}
					# for index, fname in enumerate(fnames_for_loading_scores):
					for metric_name in ['auc_pr', 'vus_pr']:
						metric_values = scoreloader.compute_metric(univarate_labels_dict.values(), weighted_scores,
															   metric=metric_name, n_jobs=8)
						metric_values_dict[metric_name] = metric_values
					for metric_name in [f'interpretability_hit_{k}_score' for k in range(1, (len(scoreloader.get_detector_names())+1)//2)]:
						metric_values = scoreloader.compute_interpretability_metric(multivariate_labels_dict.values(), weighted_contribution_scores,
																   metric=metric_name, n_jobs=8)
						metric_values_dict[metric_name] = metric_values

					# Save results
					detector_names = scoreloader.get_detector_names()
					weights_df = pd.DataFrame(weights, fnames_for_loading_scores,
											  columns=[f"weight_{x}" for x in detector_names])
					metric_results_df = pd.DataFrame(index=fnames_for_loading_scores)
					for metric_name, metric_values in metric_values_dict.items():
						metric_results_df[f"{metric_name.upper()}"] = metric_values
					results_df = pd.concat([metric_results_df, weights_df], axis=1)
					combine_detector_results_dir = combine_detector_evaluation_config.combine_detector_results_dir
					os.makedirs(combine_detector_results_dir, exist_ok=True)

					experiment_dir = os.path.join(combine_detector_results_dir)
					os.makedirs(experiment_dir, exist_ok=True)
					# filename = f"{split.replace('unsupervised_', '') if 'unsupervised' in split else 'TSB'}_{combination_method}_{k}.csv"
					filename = f"{model_name}_{window_size}_{combination_method}_{k}.csv"
					results_df.to_csv(os.path.join(experiment_dir, filename))
					print("Results saved at:", os.path.join(experiment_dir, filename))

	else:
		raise ValueError("You should provide the path to the specific splits via 'read_from_file' parameter.")
		# # Read data (single csv file or directory with csvs)
		# if '.csv' == data_path[-len('.csv'):]:
		# 	tmp_fnames = [data_path.split('/')[-1]]
		# 	data_path = data_path.split('/')[:-1]
		# 	data_path = '/'.join(data_path)
		# else:
		# 	tmp_fnames = read_files(data_path)
		#
		# # Keep specific time series if fnames is given
		# if fnames is not None:
		# 	fnames_len = len(fnames)
		# 	fnames = [x for x in tmp_fnames if x in fnames]
		# 	if len(fnames) != fnames_len:
		# 		raise ValueError("The data path does not include the time series in fnames")
		# else:
		# 	fnames = tmp_fnames
		#
		# fnames_for_loading_scores = fnames

	# return results_df


# results_ = results.sort_index()
	# results.columns = [f"{classifier_name}_{x}" for x in results.columns.values]
	#
	# # Print results
	# print(results)
	# counter = dict(Counter(results[f"{classifier_name}_class"]))
	# counter = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)}
	# print(counter)
	#
	# # Save the results
	# if path_save is not None:
	# 	file_name = os.path.join(path_save, f"{classifier_name}_preds.csv")
	# 	results.to_csv(file_name)
	#
	# return results

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
	combine_detector_evaluation_config = cfg.combine_detectors_evaluation


	eval_combine_multiple_detectors(
		combine_detector_evaluation_config=combine_detector_evaluation_config,
		# data_path=windowed_data_path,
		# model_name=model_short_name,
		# model_path=model_path,
		# model_parameters_file=model_parameters_file,
		# path_save=combine_detector_evaluation_config.path_prediction_save,
		# read_from_file=combine_detector_evaluation_config.train_test_split_file,
	)

	if combine_detector_evaluation_config.merged == True:
		merged_results_dir = combine_detector_evaluation_config.combine_detector_results_dir
		dfs = []
		for filename in os.listdir(merged_results_dir):
			pattern = re.compile(r'(?P<model_name>\w+)_(?P<window_size>\d+)_(?P<strategy>(average|vote))_(?P<top_k>\d+).csv$')
			first_match = re.match(pattern, filename)
			if first_match:
				file_path = os.path.join(merged_results_dir, filename)
				df = pd.read_csv(file_path, index_col=0)
				df.reset_index(inplace=True)
				df['dataset'] = df['index'].str.split('/', expand=True)[0]
				df['filename'] = df['index'].str.split('/', expand=True)[1]
				# df[['dataset','filename']] = df['index'].str.split('/', expand=True)[:,:2]
				df.drop(columns=['index'], inplace=True)
				model_name = first_match.group('model_name')
				window_size = first_match.group('window_size')
				combine_strategy = first_match.group('strategy')
				top_k = first_match.group('top_k')
				df['Model Selector'] = model_name+'_'+window_size
				df['Combine Method'] = combine_strategy
				df['k'] = int(top_k)
				dfs.append(df)

		merged_df = pd.concat(dfs, axis=0, ignore_index=True)
		print(f'Shape of merged results: {merged_df.shape}')
		merged_df.to_csv(os.path.join(merged_results_dir, 'merged_combine_detectors_results.csv'))
		print("Merged results saved at:", os.path.join(merged_results_dir, 'merged_combine_detectors_results.csv'))
		# train_deep_model(
		# 	data_path=train_deep_model_config.data_path,
		# 	split_per=train_deep_model_config.split_per,
		# 	seed=train_deep_model_config.seed,
		# 	read_from_file=train_deep_model_config.read_from_file,
		# 	model_name=train_deep_model_config.model_name,
		# 	model_parameters_file=train_deep_model_config.model_parameters_file,
		# 	batch_size=train_deep_model_config.batch_size,
		# 	epochs=train_deep_model_config.epochs,
		# 	eval_model=train_deep_model_config.eval_model,
		# 	path_model_save=train_deep_model_config.path_model_save,
		# 	save_done_training=train_deep_model_config.save_done_training,
		# 	path_prediction_save=train_deep_model_config.path_prediction_save,
		# 	path_save_runs=train_deep_model_config.path_save_runs,
		# )

if __name__ == "__main__":
	# parser = argparse.ArgumentParser(
	# 	prog='Evaluate deep learning models',
	# 	description='Evaluate all deep learning architectures on a single or multiple time series \
	# 		and save the results'
	# )
	
	# parser.add_argument('-d', '--data', type=str, help='path to the time series to predict', required=True)
	# parser.add_argument('-m', '--model', type=str, help='model to run', required=True)
	# parser.add_argument('-mp', '--model_path', type=str, help='path to the trained model', required=True)
	# parser.add_argument('-pa', '--params', type=str, help="a json file with the model's parameters", required=True)
	# parser.add_argument('-ps', '--path_save', type=str, help='path to save the results', default="results/raw_predictions")
	# parser.add_argument('-f', '--file', type=str, help='path to file that contains a specific split', default=None)
	#
	# args = parser.parse_args()
	main()