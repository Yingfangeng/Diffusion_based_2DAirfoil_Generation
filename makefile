training:
	python -m models.diffusion_model mdl_hyperparams/3D_aux_75.yaml > mdl_weight/training_log/3D_aux_75.log
	python -m models.diffusion_model mdl_hyperparams/3D_aux_50.yaml > mdl_weight/training_log/3D_aux_50.log
	python -m models.diffusion_model mdl_hyperparams/3D_aux_25.yaml > mdl_weight/training_log/3D_aux_25.log

	python -m models.diffusion_model mdl_hyperparams/3D_coords_conv_2d_unet_75.yaml > mdl_weight/training_log/3D_coords_conv_2d_unet_75.log
	python -m models.diffusion_model mdl_hyperparams/3D_coords_conv_2d_unet_50.yaml > mdl_weight/training_log/3D_coords_conv_2d_unet_50.log
	python -m models.diffusion_model mdl_hyperparams/3D_coords_conv_2d_unet_25.yaml > mdl_weight/training_log/3D_coords_conv_2d_unet_25.log



validation:
	python model_validation.py mdl_hyperparams/3D_coords_conv_2d_unet_75.yaml mdl_hyperparams/3D_aux_75.yaml
	python model_validation.py mdl_hyperparams/3D_coords_conv_2d_unet_50.yaml mdl_hyperparams/3D_aux_50.yaml
	python model_validation.py mdl_hyperparams/3D_coords_conv_2d_unet_25.yaml mdl_hyperparams/3D_aux_25.yaml
