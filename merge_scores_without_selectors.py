########################################################################
#
# @author : Emmanouil Sylligardos
# @when : Winter Semester 2022/2023
# @where : LIPADE internship Paris
# @title : MSAD (Model Selection Anomaly Detection)
# @component: root
# @file : merge_scores
#
########################################################################
import hydra
from omegaconf import DictConfig

from merge_scores import merge_scores_mts_without_selector



@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
	print(f'Merge scores with configuration: {cfg}')
	# if cfg.merge_score.mts == False:
	# 	merge_scores(
	# 		path=cfg.merge_score.path,
	# 		metric=cfg.merge_score.metrics[0],
	# 		save_path=cfg.merge_score_save_path,
	# 	)
	# else:
	# metrics = cfg.merge_score.metrics
	metric_for_optimization = cfg.mts_current_metric_for_optimization
	# Only detectors + Oracle + Avg Ens
	# for m in metrics:
	merge_scores_mts_without_selector(
		path=cfg.merge_score.raw_predictions_path,
		metric_for_optimization=metric_for_optimization,
		save_path=cfg.merge_score.save_path,
		mts_metrics_path=cfg.mts_current_metrics_path,
		mts_acc_tables_path=cfg.mts_current_acc_tables_path,
	)

if __name__ == "__main__":

	# parser = argparse.ArgumentParser(
	# 	prog='Merge scores',
	# 	description="Merge all models' scores into one csv"
	# )
	# parser.add_argument('-p', '--path', help='path of the files to merge')
	# parser.add_argument('-m', '--metric', help='metric to use')
	# parser.add_argument('-s', '--save_path', help='where to save the result')
	# parser.add_argument('-mts', '--mts', help='working on mts', default=False, type=bool)
	# parser.add_argument('-time', '--time-true', action="store_true", help='whether to produce time results')
	#
	# args = parser.parse_args()

	# --path = results/raw_predictions/
	# --metric = vus_roc
	# --save_path = results_mts
	# --mts = true
	# -time
	main()

