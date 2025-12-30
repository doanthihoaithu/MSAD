source activate base
conda activate tf_gpu

#python run_oracle.py
#conda activate timeeval_py38
#python run_avg_ens.py
#conda activate tf_gpu

#python create_windows_dataset_for_mts.py
#python generate_features.py

python train_rocket.py
python train_feature_based.py
python train_deep_model.py

python merge_scores.py

conda deactivate