training:
	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_100.yaml
	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_90.yaml
	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_80.yaml
	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_70.yaml
	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_60.yaml
	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_50.yaml
	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_40.yaml
	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_30.yaml
	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_20.yaml
	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_10.yaml
	python -m models.diffusion_model mdl_hyperparams/1D_params_unet_1.yaml



validation:
	python model_validation.py mdl_hyperparams/1D_params_unet_100.yaml
	python model_validation.py mdl_hyperparams/1D_params_unet_90.yaml
	python model_validation.py mdl_hyperparams/1D_params_unet_75.yaml
	python model_validation.py mdl_hyperparams/1D_params_unet_50.yaml