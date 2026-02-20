########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : generate_features
#
########################################################################
import hydra
import numpy as np
import pandas as pd
import argparse
import re
import os

from omegaconf import DictConfig
from tqdm import tqdm

from utils.data_loader import DataLoader

from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor


def generate_features(path):
	"""Given a dataset it computes the TSFresh automatically extracted 
	features and saves the new dataset (which does not anymore contain
	time series but tabular data) into one .csv in the folder of the
	original dataset

	:param path: path to the dataset to be converted
	"""
	window_size = int(re.search(r'\d+$', path).group())

	# Create name of new dataset
	dataset_name = [x for x in path.split('/') if str(window_size) in x][0]
	new_name = f"TSFRESH_{dataset_name}.csv"

	# Load datasets
	dataloader = DataLoader(path)
	datasets = dataloader.get_dataset_names()
	df = dataloader.load_df(datasets)
	
	# Divide df
	label_columns = [f for f in df.columns if 'label_by_' in f]
	# Multiple labels [2, n] with using both auc_pr, interpretabilaty for labeling
	label_by_metric_df = df[label_columns].copy()
	df = df.drop(label_columns, axis=1)


	#TODO update for extract features for MTS
	# x shape = [k, window_size, n_features] based on windowing dataset before flatten
	# x = df.to_numpy()[:, np.newaxis]
	x = df.to_numpy().reshape(df.shape[0], window_size,-1)
	index = df.index

	# Setup the TSFresh feature extractor (too costly to use any other parameter)
	fe = TSFreshFeatureExtractor(
		default_fc_parameters="minimal",  # 'minimal' or 'efficient' or 'comprehensive'
		show_warnings=False, 
		n_jobs=-1
	)
	
	# Compute features
	X_transformed = fe.fit_transform(x)

	# Create new dataframe
	X_transformed.index = index
	X_transformed = pd.merge(label_by_metric_df, X_transformed, left_index=True, right_index=True)
	
	# Save new features
	X_transformed.to_csv(os.path.join(path, new_name))

@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
	if cfg.generate_features.mode == "single":
		generate_features(
			path=cfg.generate_features.path,
		)
	else:
		window_sizes = cfg.generate_features.window_sizes
		for window_size in tqdm(window_sizes, desc="Generating features for different window sizes", total=len(window_sizes)):
			path = cfg.generate_features.path_template.format(current_window_size=window_size)
			generate_features(
				path=path,
			)


if __name__ == "__main__":
	# parser = argparse.ArgumentParser(
	# 	prog='generate_features',
	# 	description='Transform a dataset of time series (of equal length) to tabular data\
	# 	with TSFresh'
	# )
	# parser.add_argument('-p', '--path', type=str, help='path to the dataset to use')
	#
	# args = parser.parse_args()
	# --path = data / mts / settings_two / settings_two_32
	main()
