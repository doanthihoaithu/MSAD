import os

import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.utils import get_project_root


def train_test_split_customized(filenames, labels, dataset, data_dir, train_ratio=0.6):
    filenames = ["/".join(f.split('/')[-2:]) for f in filenames]
    # np.random.shuffle(filenames)
    # split_index = int(len(filenames) * train_ratio)
    # train_filenames = filenames[:split_index]
    # test_filenames = filenames[split_index:]
    train_filenames, test_filenames, train_labels, test_labels = train_test_split(filenames, labels, train_size=train_ratio)
    print(f'Train set size: {len(train_filenames)}, Test set size: {len(test_filenames)}')
    print(f'Unique labels in train set: {len(set(train_labels))}, Unique labels in test set: {len(set(test_labels))}')
    while len(set(train_labels)) < len(set(test_labels)):
        train_filenames, test_filenames, train_labels, test_labels = train_test_split(filenames, labels,
                                                                                      train_size=train_ratio)
        print(f'Train set size: {len(train_filenames)}, Test set size: {len(test_filenames)}')
        print(
            f'Unique labels in train set: {len(set(train_labels))}, Unique labels in test set: {len(set(test_labels))}')

    train_filenames_df = pd.DataFrame(train_filenames, columns=['train_set'])
    test_filenames_df = pd.DataFrame(test_filenames, columns=['val_set'])
    combined_df = pd.concat([train_filenames_df, test_filenames_df], axis=1, ignore_index=False).T
    combined_df.columns = [f'C{i}' for i in range(combined_df.shape[1])]
    # for window_size in window_sizes:
    save_dir = os.path.join(os.path.dirname(data_dir), 'supervised_splits')
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir,'train_test_split.csv')
    combined_df.to_csv(save_file, header=True, index=True)
    print("Saved train-test split to:", save_file)

@hydra.main(config_path="./conf", config_name="config.yaml")
def main(config: DictConfig):
    print(config)
    # project_root_dir = get_project_root()
    current_working_dataset = config.mts_running_dataset
    metric_for_labeling_windows = config.metric_for_labeling_windows
    # msad_mts_working_data_dir = os.path.join(project_root_dir, config.msad_mts_working_data_dir)
    msad_mts_data_dir = config.mts_current_data_path
    metrics_dir = config.mts_current_metrics_path
    mts_supported_detectors = config.mts_supported_detectors
    print(f'mts_supported_detectors: {mts_supported_detectors}')

    auc_pr_dfs = []
    # vus_roc_dfs = []
    for alg in os.listdir(metrics_dir):
        if alg in mts_supported_detectors:
            # print(alg)
            auc_pr_path = os.path.join(metrics_dir, alg, f'{metric_for_labeling_windows.upper()}.csv')
            auc_pr_df = pd.read_csv(auc_pr_path, index_col=0)
            auc_pr_dfs.append(auc_pr_df)

    combined_auc_pr_df = pd.concat(auc_pr_dfs, axis=1)
    combined_auc_pr_df.index.name = 'filename'
    combined_auc_pr_df['dataset'] = current_working_dataset
    combined_auc_pr_df.dropna(inplace=True)
    labels = combined_auc_pr_df[mts_supported_detectors].idxmax(axis=1)
    print(pd.DataFrame(labels).value_counts())
    # for window_size in window_sizes:
    train_test_split_customized(combined_auc_pr_df.index.str.replace('.out', '.out.csv').tolist(), labels, current_working_dataset, msad_mts_data_dir,
                                train_ratio=0.5)


if __name__ == '__main__':
    main()