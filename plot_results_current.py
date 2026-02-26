import os.path

import hydra
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from plotting_constants import methods_colors, combined_detector_methods, methods_feature, methods_ts, methods_sit, \
    methods_conv, template_names
from utils.metrics_loader import MetricsLoader
from utils.utils import get_project_root
import seaborn as sns


def plot_result_boxplot_dataset(detectors, final_names, measure_names, results_dir, save_dir, test_filenames, all_methods_ens):

    plt.rcParams.update({'font.size': 18})
    # plt.figure(figsize=(10, 5*len(measure_names)))
    plt.grid(color='k', linestyle='--', linewidth=1, alpha=0.2)

    figure, axis = plt.subplots(figsize=(10, 5*len(measure_names)), nrows=len(measure_names), ncols=1)

    project_root_dir = get_project_root()
    test_filenames = [f.split('/')[1].replace('.out.csv', '.out') for f in test_filenames]
    for i, measure_name in enumerate(measure_names):
        total_of_metric_file_name = f'current_accuracy_{measure_name}.csv'
        total_of_metric_file_path = os.path.join(project_root_dir, results_dir, total_of_metric_file_name)

        df = pd.read_csv(total_of_metric_file_path)
        df = df[df['filename'].isin(test_filenames)]
        # df = pd.read_csv(
        #     f'../../results_mts/label_by_{mts_current_metric_for_optimization}/merged_scores/{current_dataset}/total_accuracy_{measure_name}.csv')
        # df = pd.merge(df, df_average, on=['dataset', 'filename'], how='left')
        # df = pd.merge(df, df_vote, on=['dataset', 'filename'], how='left')
        score_columns = [c for c in df.columns if c.endswith('_score')]
        new_columns = {c: c[:-6] for c in score_columns}
        # print(new_columns)
        # print(score_columns)
        df.rename(new_columns, axis=1, inplace=True)
        # df.columns = df.columns.str[:-5]
        print(df.columns, df.shape)

        # Calculate the mean performance for each Model Selector (MS)
        method_means = df[[x for x in all_methods_ens if x in df.columns]].median()
        print(df.columns)
        print(all_methods_ens)
        print('Selectors:',method_means )
        best_ms = method_means.idxmax() # Best MS may differ since results are may slightly vary (although distributions are solid)
        # best_ms = 'xxx'
        #     best_ms = 'resnet_1024' # Best model selector at the time publishing of our paper
        print(f"Best Model Selector (MS) for VUS is: {best_ms}--->{final_names[best_ms]}")

        best_vus_pr_v1 = None
        if measure_name == 'VUS_PR':
            best_vus_pr_v1 = best_ms
            # best_vus_pr_v2 = best_ms_v2
            # best_vus_pr_v2_list = selected_best_ms_combine_raw_names
            print('best_ms_v1 for VUS_PR:', best_vus_pr_v1)
            # print('best_ms_v2 for VUS_PR:', best_vus_pr_v2)
        else:
            # best_ms_v2 = best_vus_pr_v2
            best_ms = best_vus_pr_v1
            # selected_best_ms_combine_raw_names = best_vus_pr_v2_list

        # old_method_order = ['OCSVM','POLY','LSTM','CNN','HBOS','PCA','IFOREST','AE','LOF','IFOREST1','MP','NORMA']
        old_method_order = df[detectors].median().sort_values(ascending=False).index.tolist()[::-1]
        my_pal = {method: methods_colors["detectors"] for method in old_method_order}
        my_pal = {**my_pal, **{"Avg Ens": methods_colors["avg_ens"],
                               best_ms: methods_colors["best_ms"],
                               'Oracle': methods_colors["oracle"]}}

        tmp_methods = old_method_order + ['Avg Ens', best_ms, 'Oracle']
        bplot = sns.boxplot(data=df[tmp_methods], palette=my_pal, showfliers=False, saturation=1, whis=0.241289844, ax=axis[i])

        xticks_labels = []
        for x in tmp_methods:
            if x != best_ms:
                xticks_labels.append(final_names[x])
            else:
                xticks_labels.append(final_names['best_ms'])

        axis[i].set_xticks(list(range(len(xticks_labels))), xticks_labels, rotation=45)
        axis[i].set_ylabel(final_names[measure_name])

        axis[i].axvline(9.5, color='black', linestyle='--')
        # axis[i].tight_layout()
        # axis[i].sa(os.path.join(save_dir, f'all_data_{measure_name}.png'), transparent=True)
        #     plt.savefig(figure_path.format('1_intro_fig_1'), transparent=True)
        # plt.show()
    plt.tight_layout()
    figure.savefig(os.path.join(save_dir, f'current_data_all_measures.png'), transparent=True)
    plt.close()


@hydra.main(config_path="conf", config_name="config.yaml")
def main(config: DictConfig) -> None:
    print(config)

    current_dataset = config.mts_running_dataset
    mts_current_metric_for_optimization = config.mts_current_metric_for_optimization
    print(current_dataset, mts_current_metric_for_optimization)
    project_root_dir = get_project_root()
    results_dir = config.merge_score.save_path
    mts_current_metrics_path = os.path.join(project_root_dir, config.mts_current_metrics_path)
    metricsloader = MetricsLoader(mts_current_metrics_path)
    train_test_split_file = os.path.join(project_root_dir, config.split_train_test.file)
    train_test_split_df = pd.read_csv(train_test_split_file, index_col=0)
    test_filenames = train_test_split_df.loc['val_set'].values
    print(test_filenames)

    measure_names = metricsloader.get_names()


    detectors = config.mts_supported_detectors

    # old_methods = ['cblof', 'auto_encoder', 'copod', 'denoising_auto_encoder', 'encdec_ad', 'hbos', 'omni_anomaly',
    #                'random_black_forest', 'tran_ad', 'mtad_gat']
    #
    Base_methods = ['Avg Ens', 'Oracle']

    all_length = config.supported_window_sizes
    #
    all_methods_ens = [meth.format(length) for meth in methods_conv for length in all_length]
    all_methods_ens += [meth.format(length) for meth in methods_sit for length in all_length]
    all_methods_ens += [meth.format(length) for meth in methods_ts for length in all_length]
    all_methods_ens += [meth.format(length) for meth in methods_feature for length in all_length]
    # all_methods_ens += combined_detector_methods

    # Keep only the methods that exist in the results you read
    all_methods = detectors + Base_methods + all_methods_ens
    # all_methods = [x for x in all_methods if x in df.columns]

    # Create a list of all different classes of methods
    split = [x.rsplit('_', 1)[0] for x in all_methods]
    used = set()
    # all_methods_class = [x for x in split if x not in used and (used.add(x) or True)]


    final_names = {}
    for length in all_length:
        for key, value in template_names.items():
            if '{}' in key:
                new_key = key.format(length)
                new_value = value.format(length)
                final_names[new_key] = new_value
            else:
                final_names[key] = value
    print(final_names)

    plot_save_dir = os.path.join(project_root_dir, config.merge_score.save_path, 'plots')
    os.makedirs(plot_save_dir, exist_ok=True)

    plot_result_boxplot_dataset(detectors, final_names, measure_names, results_dir, plot_save_dir, test_filenames, all_methods_ens)

if __name__ == '__main__':
    main()
