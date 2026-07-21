training:

# 	python -m models.diffusion_model mdl_hyperparams/3D_aux.yaml > mdl_weight/training_log/3D_aux_mar_03.log
# 	python -m models.diffusion_model mdl_hyperparams/3D_aux_mlp.yaml > mdl_weight/training_log/3D_aux_mlp_15_July.log
	python -m models.diffusion_model mdl_hyperparams/3D_coords_conv_2d_unet.yaml > mdl_weight/training_log/3D_coords_conv_2d_unet_15_July.log

	
	



validation:

# 	python model_validation.py mdl_hyperparams/1D_params_mlp.yaml mdl_hyperparams/3D_aux_mlp.yaml
	python model_validation.py mdl_hyperparams/3D_coords_conv_2d_unet.yaml mdl_hyperparams/3D_aux_mlp.yaml
