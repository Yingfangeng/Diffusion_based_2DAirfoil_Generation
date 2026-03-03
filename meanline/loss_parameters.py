from dataclasses import dataclass

@dataclass
class LossModelParameters:
    # clearance: float = 0.6
    # clearance_exp: float = 0.5
    # blade_loading: float = 0.05
    # blade_loading_coeff: float = 0.75
    # blade_loading_exp: float = 2.0
    # blade_loading_offset: float = 0.0
    # incidence: float = 0.4
    # entrance_diffusion_factor: float = 0.4
    # entrance_diffusion_stall_factor: float = 0.5
    # recirculation: float = 0.02
    # mixing: float = 1.0
    # mixing_wake: float = 0.25
    # mixing_bstar: float = 1.0
    # skin_friction: float = 1.0
    # sf_coeff_lam: float = 3.7
    # sf_coeff_turb: float = 0.0102
    # sf_exp_adjust: float = 1.0
    # leakage: float = 1.0
    # disk_friction: float = 1.0
    # shock: float = 0.2
    # choke_1: float = 1.0
    # choke_2: float = 0.05
    # choke_3: float = 7.0
    # choke_limit: float = 1.0
    # choke_scale: float = 10
    # choke_offset: float = 1.1
    # vd_factor: float = 0.01
    # vd_power: float = 0.20
    # slip_factor_factor: float = 0.26
    # slip_factor_power: float = -0.10
    # slip_factor_c1: float = 0.0
    # slip_factor_c2: float = 0.0
    # slip_factor_c3: float = 0.0
    # slip_factor_c4: float = 0.0



    # CALIBRATED
    clearance: float = 0.002569
    clearance_exp: float = 0.573978
    blade_loading: float = 0.089815
    blade_loading_coeff: float = 0.920631
    blade_loading_exp: float = 1.000000
    blade_loading_offset: float = -0.118382
    incidence: float = 0.089662
    entrance_diffusion_factor: float = 1.000000
    entrance_diffusion_stall_factor: float = 0.879654
    recirculation: float = 0.025388
    mixing: float = 7.237361
    mixing_wake: float = 0.113048
    mixing_bstar: float = 0.628064
    skin_friction: float = 1.490142
    sf_coeff_lam: float = 3.935176
    sf_coeff_turb: float = 0.008785
    sf_exp_adjust: float = 1.172279
    leakage: float = 0.144676
    disk_friction: float = 0.622096
    shock: float = 0.428346
    choke_1: float = 0.600776
    choke_2: float = 0.038735
    choke_3: float = 3.202012
    choke_limit: float = 0.993673
    choke_scale: float = 14.465246
    choke_offset: float = 1.011220
    slip_factor_factor: float = 0.378377
    slip_factor_power: float = -0.027388
    slip_factor_c1: float = -0.126604
    slip_factor_c2: float = -1.407397
    slip_factor_c3: float = 1.022075
    slip_factor_c4: float = -4.588839
    vd_factor: float = 0.01
    vd_power: float = 0.20