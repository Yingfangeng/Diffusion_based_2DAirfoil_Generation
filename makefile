training:
# 	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_100.yaml
# 	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_90.yaml
# 	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_80.yaml
# 	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_70.yaml
# 	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_60.yaml
# 	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_50.yaml
# 	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_40.yaml
# 	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_30.yaml
# 	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_20.yaml
# 	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_10.yaml
# 	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_5.yaml > mdl_weight/training_log/5_percent_data_training.log
# 	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_1.yaml
# 	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_0.5.yaml > mdl_weight/training_log/0.5_percent_data_training.log
# 	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_0.1.yaml > mdl_weight/training_log/0.1_percent_data_training.log

	python -m models.diffusion_model mdl_hyperparams/3D_coords_unet.yaml > mdl_weight/training_log/3D_coordinates_200_epochs.log



validation:
# 	python model_validation.py mdl_hyperparams/1D_params_unet_100.yaml
# 	python model_validation.py mdl_hyperparams/1D_params_unet_90.yaml
# 	python model_validation.py mdl_hyperparams/1D_params_unet_80.yaml
# 	python model_validation.py mdl_hyperparams/1D_params_unet_70.yaml
# 	python model_validation.py mdl_hyperparams/1D_params_unet_60.yaml
# 	python model_validation.py mdl_hyperparams/1D_params_unet_50.yaml
# 	python model_validation.py mdl_hyperparams/1D_params_unet_40.yaml
# 	python model_validation.py mdl_hyperparams/1D_params_unet_30.yaml
# 	python model_validation.py mdl_hyperparams/1D_params_unet_20.yaml
# 	python model_validation.py mdl_hyperparams/1D_params_unet_10.yaml
	python model_validation.py mdl_hyperparams/1D_params_unet_5.yaml
# 	python model_validation.py mdl_hyperparams/1D_params_unet_1.yaml
# 	python model_validation.py mdl_hyperparams/1D_params_unet_0.5.yaml
# 	python model_validation.py mdl_hyperparams/1D_params_unet_0.1.yaml

