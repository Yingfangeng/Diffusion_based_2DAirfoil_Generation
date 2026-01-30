from dataclasses import dataclass

@dataclass
class LossModelParameters:
    clearance: float = 0.6
    clearance_exp: float = 0.5
    blade_loading: float = 0.05
    blade_loading_coeff: float = 0.75
    blade_loading_exp: float = 2.0
    blade_loading_offset: float = 0.0
    incidence: float = 0.4
    entrance_diffusion_factor: float = 0.4
    entrance_diffusion_stall_factor: float = 0.5
    recirculation: float = 0.02
    mixing: float = 1.0
    mixing_wake: float = 0.25
    mixing_bstar: float = 1.0
    skin_friction: float = 1.0
    sf_coeff_lam: float = 3.7
    sf_coeff_turb: float = 0.0102
    sf_exp_adjust: float = 1.0
    leakage: float = 1.0
    disk_friction: float = 1.0
    shock: float = 0.2
    choke_1: float = 1.0
    choke_2: float = 0.05
    choke_3: float = 7.0
    choke_limit: float = 1.0
    choke_scale: float = 10
    choke_offset: float = 1.1
    vd_factor: float = 0.01
    vd_power: float = 0.20
    slip_factor_factor: float = 0.26
    slip_factor_power: float = -0.10
    slip_factor_c1: float = 0.0
    slip_factor_c2: float = 0.0
    slip_factor_c3: float = 0.0
    slip_factor_c4: float = 0.0