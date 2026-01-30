import math
import CoolProp.CoolProp as CP

from meanline.loss_parameters import LossModelParameters

class Centrifugal_Compressor_Losses:
    
    
    def __init__(self, s, b_2, r_1_hub, r_1_tip, r_2, U_2, C_2, nu_01, beta_1_tip, beta_1_hub, beta_2, 
                 N_blades, L_z, W_2, W_1_tip, Euler_work, W_u_1, alpha_2, rho_2, C_theta_2, r_1, 
                 C_u_1, b_1, m_dot_1, A_2, gamma, R, beta_b2, T_2, rho_1, T_1, A_1, beta_b1, t, rho_01, 
                 T_01, U_1, W_1, T_02, C_1, nu_2, beta_1, W_1_hub, alpha_1, U_1_tip, omega, P_2, 
                 mu_2, beta_b1_tip, beta_b1_hub, P_1, C_p, slip_factor, n_splitter_blades, A_th, loss_params=None):
        """
        Initialise parameters needed for centrifugal impeller loss calculations
        """
        self.s = s
        self.b_2 = b_2
        self.r_1_hub = r_1_hub
        self.r_1_tip = r_1_tip
        self.r_2 = r_2
        self.U_2 = U_2
        self.C_2 = C_2
        self.nu_01 = nu_01
        self.beta_1_tip = beta_1_tip
        self.beta_1_hub = beta_1_hub
        self.beta_2 = beta_2
        self.N_blades = N_blades
        self.L_z = L_z
        self.W_2 = W_2
        self.W_1_tip = W_1_tip
        self.Euler_work = Euler_work
        self.W_u_1 = W_u_1
        self.alpha_2 = alpha_2
        self.rho_2 = rho_2
        self.C_theta_2 = C_theta_2
        self.r_1 = r_1
        self.C_u_1 = C_u_1
        self.b_1 = b_1
        self.m_dot_1 = m_dot_1
        self.A_2 = A_2
        self.gamma = gamma
        self.R = R
        self.beta_b2 = beta_b2
        self.T_2 = T_2
        self.rho_1 = rho_1
        self.T_1 = T_1
        self.A_1 = A_1
        self.beta_b1 = beta_b1
        self.t = t
        self.rho_01 = rho_01
        self.T_01 = T_01
        self.U_1 = U_1
        self.W_1 = W_1
        self.T_02 = T_02
        self.C_1 = C_1
        self.nu_2 = nu_2
        self.beta_1 = beta_1
        self.W_1_hub = W_1_hub
        self.alpha_1 = alpha_1
        self.U_1_tip = U_1_tip
        self.omega = omega
        self.P_2 = P_2
        self.mu_2 = mu_2
        self.beta_b1_tip = beta_b1_tip
        self.beta_b1_hub = beta_b1_hub
        self.P_1 = P_1
        self.C_p = C_p
        self.slip_factor = slip_factor
        self.n_splitter_blades = n_splitter_blades
        self.A_th = A_th

        # get the loss parameters for all loss modes
        if loss_params is None:
            loss_params = LossModelParameters()
        self.loss_params = loss_params
        



##################################### INTERNAL LOSSES ##################################################################
########################################################################################################################




    def clearance_loss(self):
        """
        Jansen clearance loss with flexible exponent
        """
        clearance_method = 'Jansen'
        
        if clearance_method == 'Jansen':
            factor = self.loss_params.clearance
            exponent = self.loss_params.clearance_exp  # NEW (default 0.5)
            
            term = ((4*math.pi)/(self.b_2*self.N_blades)) * \
                ((self.r_1_tip**2 - self.r_1_hub**2)/(self.r_2 - self.r_1_tip)) * \
                self.C_theta_2 * self.C_1 / (1+self.rho_2/self.rho_1)
            
            Loss = self.m_dot_1 * factor * (self.s/self.b_2) * \
                (self.C_theta_2) * (term**exponent)
        
        elif clearance_method == 'Rodgers':
            factor = self.loss_params.clearance
            Loss = self.m_dot_1 * factor * (self.s/self.b_2) * (self.U_2**2)
        
        elif clearance_method == 'Krylov':
            factor = self.loss_params.clearance
            Loss = self.m_dot_1 * factor * (self.s/self.b_2) * \
                ((self.r_1_hub + self.r_1_tip)/(2*self.r_2) - 0.275) * (self.U_2**2)
        
        else:
            Loss = 0
        
        return Loss




    
    def skin_friction_loss(self):
        """
        Jansen skin friction with calibratable Reynolds correlation
        """
        skin_friction_method = 'Jansen'
        
        if skin_friction_method == 'Jansen':
            # Mean relative velocity squared
            W_bar_sq = 0.5*(self.W_1**2 + self.W_2**2)
            
            # Impeller Flow Length
            L_b = (math.pi/4)*(2*self.r_2 - (self.r_1_hub+self.r_1_tip) - self.b_2 + 2*self.L_z) / \
                (math.sin(math.pi/2-abs(self.beta_b1)) + math.sin(math.pi/2-abs(self.beta_b2)))
            
            # Hydraulic Diameter
            D_hyd = (2*self.r_2*math.sin(math.pi/2-abs(self.beta_2))) / \
                    ((self.N_blades-0.5*self.n_splitter_blades)/math.pi + 
                    (2*self.r_2*math.sin(math.pi/2-abs(self.beta_2)))/(self.b_2)) + \
                    (0.5*(2*self.r_1_tip + 2*self.r_1_hub) * 
                    (0.5*(math.sin(math.pi/2-abs(self.beta_1_tip))+
                        math.sin(math.pi/2-abs(self.beta_1_hub))))) / \
                    ((self.N_blades - self.n_splitter_blades)/math.pi + 
                    ((self.r_1_tip+self.r_1_hub)/(self.r_1_tip-self.r_1_hub))*(0.5)*
                    (math.sin(math.pi/2-abs(self.beta_1_tip))+
                    math.sin(math.pi/2-abs(self.beta_1_hub))))
            
            Re = self.U_2*self.r_2/(self.nu_2)
            
            # NEW: Calibratable parameters
            Re_trans = 3e5
            coeff_lam = self.loss_params.sf_coeff_lam     # NEW (default 3.7)
            coeff_turb = self.loss_params.sf_coeff_turb   # NEW (default 0.0102)
            exp_adjust = self.loss_params.sf_exp_adjust   # NEW (default 1.0)
            
            if Re < Re_trans:
                C_f = coeff_lam * ((self.s/self.r_2)**(0.1)) * ((Re)**(-0.5*exp_adjust))
            else:
                C_f = coeff_turb * ((self.s/self.r_2)**(0.1)) * ((Re)**(-0.2*exp_adjust))
            
            Loss = 2 * C_f * (L_b/D_hyd) * W_bar_sq * self.m_dot_1 * \
                self.loss_params.skin_friction
        
        else:
            Loss = 0
        
        return Loss
        
        
    
    

    def blade_loading_loss(self):
        """
        Enhanced Coppage blade loading with flexible exponent and coefficient
        """
        blade_loading_method = 'Coppage'
        
        if blade_loading_method == 'Coppage':
            factor = self.loss_params.blade_loading
            coeff = self.loss_params.blade_loading_coeff  # NEW (default 0.75)
            exponent = self.loss_params.blade_loading_exp  # NEW (default 2.0)
            offset = self.loss_params.blade_loading_offset  # NEW (default 0.0)
            
            # Diffusion factor calculation
            D_f = 1 - (self.W_2)/(self.W_1_tip) + \
                (coeff * self.Euler_work * self.W_2 / self.m_dot_1) / \
                (((self.N_blades)/math.pi) * (1 - self.r_1_tip/self.r_2) + 
                (2*self.r_1_tip)/(self.r_2)) / \
                (self.W_1_tip * (self.U_2**2))
            
            # Allow offset for better calibration flexibility
            Loss = factor * ((D_f + offset)**exponent) * (self.U_2**2) * self.m_dot_1
        
        elif blade_loading_method == 'Aungier':
            L_b = (math.pi/4)*(2*self.r_2 - (self.r_1_hub+self.r_1_tip) - self.b_2 + 2*self.L_z) / \
                (math.sin(math.pi/2-abs(self.beta_b1)) + math.sin(math.pi/2-abs(self.beta_b2)))
            delta_w = (2*math.pi*(2*self.r_2)*self.C_theta_2)/(self.N_blades*L_b)
            Loss = (self.m_dot_1*(delta_w**2))/48
        
        else:
            Loss = 0
        
        return Loss
        
        
        


        
    
    def incidence_loss(self):
        """
        Function to calculate the incidence loss in the centrifugal compressor impeller. The fluid near
        the blade will get an instantaneous speed shift to match the inlet angle. This causes fluid 
        separation and loss in the tangential velocity component, resulting in a reduction in kinetic 
        energy. This is known as incidence loss. The formulations are:

        1. Aungier
        2. Conrad
        3. Galvas
        """

        incidence_method = 'Aungier'

        ###############################################################################################
        ######################### COMMENTS: ###########################################################
        # The Aungier one is very good
        ################################################################################################
        ################################################################################################

        ######################## AUNGIER ###############################################################

        if incidence_method == 'Aungier':
            factor = self.loss_params.incidence
            Loss = factor*self.m_dot_1*((self.W_1 - self.C_1*math.cos(self.alpha_1)/math.cos(abs(self.beta_b1)))**2)
            # print('LOSS:', Loss, 'MDOT', self.m_dot_1)

        ####################### CONRAD #################################################################

        elif incidence_method == 'Conrad':
            f_inc = 0.5
            Loss = f_inc*self.m_dot_1*(self.W_u_1**2)/2


        ######################## GALVAS ############################################################### 

        elif incidence_method == 'Galvas':
            M_w1 = self.W_1/((self.gamma*self.R*self.T_1)**0.5)
            beta_opt = math.arcsin((math.sqrt(self.gamma*(M_w1**2) + 2*M_w1 + 3) - math.sqrt(self.gamma * (M_w1**2) - 2*M_w1 + 3)) / (2*M_w1))
            W_l = self.W_1*math.sin(abs(beta_opt - self.beta_1))
            Loss = (self.m_dot_1*(W_l**2))/2

        else:

            Loss = 0

        return Loss
    










    def entrance_diffusion_loss(self):
        """
        Calculate entrance diffusion loss according to Aungier (2000) / Kosuge (1982).
        
        Physical Basis:
        ---------------
        Flow must decelerate from inducer throat (minimum area, high velocity) to 
        blade leading edge. This diffusion process causes losses. At high flow rates,
        when the deceleration ratio W_1_tip/W_th exceeds 1.75, inducer stall occurs
        with dramatically increased losses.
        
        Equations (from Aungier 2000, Kosuge 1982):
        --------------------------------------------
        Normal operation (W_1_tip/W_th ≤ 1.75):
            ΔH_dif = factor × (W_1 - W_th)² - ΔH_inc
            
        Inducer stall (W_1_tip/W_th > 1.75):
            ΔH_dif = max(normal, stall_factor × (W_1_tip - 1.75×W_th)² - ΔH_inc)
            
        If ΔH_dif < 0, then ΔH_dif = 0
        
        References:
        -----------
        - Aungier, R.H. (2000). "Centrifugal Compressors", ASME Press, Eq. 31-33
        - Kosuge, H., et al. (1982). ASME J. Eng. Power, Vol. 104, pp. 782-787
        
        """
        
        # ========================================================================
        # STEP 1: Calculate throat relative velocity (simplified method)
        # ========================================================================
        
        # Throat geometry
        A_th = self.A_th  # Throat area (already provided)
        r_th = 0.5 * (self.r_1_hub + self.r_1_tip)  # Throat radius (geometric mean)
        
        # Throat static conditions (simplified - good enough for calibration)
        # Assuming small pressure and temperature drop from inlet to throat
        T_th = self.T_1 * 0.98   # ~2% temperature drop
        P_th = self.P_1 * 0.95   # ~5% pressure drop
        
        # Throat density
        rho_th = P_th / (self.R * T_th)
        
        # Throat meridional velocity from continuity
        if A_th > 1e-10:  # Avoid division by zero
            Cm_th = self.m_dot_1 / (rho_th * A_th)
        else:
            # Fallback: use inlet velocity
            Cm_th = self.C_1
            # print("[WARNING] Throat area too small, using inlet velocity")
        
        # Throat tangential velocity (blade speed at throat)
        U_th = self.omega * r_th
        
        # Throat relative velocity (assuming no prewhirl at throat)
        # W_th = sqrt(Cm_th² + U_th²)
        W_th = math.sqrt(Cm_th**2 + U_th**2)
        
        # ========================================================================
        # STEP 2: Get blade leading edge velocities (already computed elsewhere)
        # ========================================================================
        
        W_1_mean = self.W_1      # Mean relative velocity at blade LE
        W_1_tip = self.W_1_tip   # Tip (shroud) relative velocity at blade LE
        
        # ========================================================================
        # STEP 3: Get incidence loss (convert from power to specific enthalpy)
        # ========================================================================
        
        # Calculate incidence loss in power [W]
        incidence_loss_power = self.incidence_loss()
        
        # Convert to specific enthalpy [J/kg]
        if self.m_dot_1 > 1e-10:
            Delta_H_inc = incidence_loss_power / self.m_dot_1
        else:
            Delta_H_inc = 0.0
        
        # ========================================================================
        # STEP 4: Calculate deceleration ratio and check for stall
        # ========================================================================
        
        # Get calibration parameters
        factor_normal = self.loss_params.entrance_diffusion_factor          # Default: 0.4
        factor_stall = self.loss_params.entrance_diffusion_stall_factor     # Default: 0.5
        # stall_threshold = self.loss_params.entrance_diffusion_stall_ratio   # Default: 1.75
        stall_threshold = 1.75

        
        # Deceleration ratio (Kosuge criterion)
        if W_th > 1e-6:  # Avoid division by zero
            deceleration_ratio = W_1_tip / W_th
        else:
            # If throat velocity is essentially zero (shouldn't happen)
            deceleration_ratio = 1.0
            W_th = 1e-6  # Set small value to prevent numerical issues
        
        # ========================================================================
        # STEP 5: Calculate entrance diffusion loss
        # ========================================================================
        
        # Normal operation loss (Equation 31 from Aungier)
        velocity_diff = W_1_mean - W_th
        Delta_H_dif_normal = factor_normal * (velocity_diff**2) - Delta_H_inc
        
        # Check if inducer stall condition is met (Equation 33)
        if deceleration_ratio > stall_threshold:
            # INDUCER STALL REGIME (Equation 32)
            stall_term = W_1_tip - stall_threshold * W_th
            Delta_H_dif_stall = factor_stall * (stall_term**2) - Delta_H_inc
            
            # Take maximum of normal and stall formulations
            Delta_H_dif = max(Delta_H_dif_normal, Delta_H_dif_stall, 0.0)
            
        else:
            # NORMAL OPERATION REGIME
            Delta_H_dif = max(Delta_H_dif_normal, 0.0)
        
        # Ensure loss is non-negative (per Aungier)
        Delta_H_dif = max(0.0, Delta_H_dif)
        
        # ========================================================================
        # STEP 6: Convert specific enthalpy loss to power loss
        # ========================================================================
        
        Loss_power = self.m_dot_1 * Delta_H_dif  # [W]
        
        return Loss_power








    
    
    


    def shock_loss(self):
        """
        Compute shock loss total enthalpy drop. Formulations are by:

        1. Aungier
        2. Whitefield and Baines
        """

        shock_method = 'Aungier'

        if shock_method == 'Aungier':

            M_w1_tip = (self.W_1_tip)/((self.gamma*self.R*self.T_1)**0.5)

            phi = (self.C_2*math.cos(abs(self.alpha_2)))/(self.U_2)

            I_b = self.C_theta_2/self.U_2 - self.U_1*self.C_u_1/(self.U_2**2)

            L_b = (math.pi/8)*( 2*self.r_2 - (self.r_1_tip+self.r_1_hub) - self.b_2 + 2*self.L_z )*( 2/( (math.cos(self.beta_1_tip)+math.cos(self.beta_1_hub))/2 + math.cos(self.beta_2) ) )
        
            D_w = 2*math.pi*2*self.r_2*self.U_2*I_b/((self.N_blades-self.n_splitter_blades)*L_b)

            W_max = (self.W_1+self.W_2+D_w)/2

            W_star = (self.gamma*self.R*self.T_1)**0.5

            M_cr = M_w1_tip*W_star/W_max

            if M_w1_tip>M_cr:
                
                factor = self.loss_params.shock

                Loss = factor*(((M_w1_tip-M_cr)*W_max)**2)*self.m_dot_1*0.5
            
            else:
                Loss = 0 
        
        else:
                Loss = 0 

        return Loss


    
    
    
    def mixing_loss(self):
        """
        Johnston & Dean mixing loss with calibratable wake parameters
        """
        mixing_method = 'Johnston_Dean'
        
        if mixing_method == 'Johnston_Dean':
            epsilon_wake = self.loss_params.mixing_wake   # NEW (default 0.25)
            b_star = self.loss_params.mixing_bstar        # NEW (default 1.0)
            factor = self.loss_params.mixing
            
            Loss = (1/(1+(math.tan(self.alpha_2))**2)) * \
                (((1-epsilon_wake-b_star)/(1-epsilon_wake))**2) * \
                (0.5) * ((self.C_2)**2)
            Loss = self.m_dot_1 * Loss * factor
        
        else:
            Loss = 0
        
        return Loss
    
    
    



    
    def choke_loss(self):
        """
        Enhanced choke loss with calibratable scaling
        """
        self.a_01 = (self.gamma*self.R*self.T_01)**0.5
        A_th_star = (self.m_dot_1) / \
                    ((self.rho_01*self.a_01)*
                    (((2+(self.gamma-1)*((self.U_1**2)/(self.a_01**2)))/
                    (self.gamma+1))**((self.gamma+1)/(2*(self.gamma-1)))))
        
        C_r_1 = ((self.A_1*math.sin(math.pi/2-abs(self.beta_b1_tip))/self.A_th)**0.5)
        C_r_lim = 1 - ((self.A_1*math.sin(math.pi/2-abs(self.beta_b1_tip))/
                        self.A_th - 1)**2)
        C_r = min(C_r_1, C_r_lim)
        
        factor1 = self.loss_params.choke_1
        factor2 = self.loss_params.choke_2
        factor3 = self.loss_params.choke_3
        factor4 = self.loss_params.choke_limit
        scale = self.loss_params.choke_scale    # NEW (default 10)
        offset = self.loss_params.choke_offset  # NEW (default 1.1)
        
        x_ch = scale * (offset - factor4 * (C_r * self.A_th) / A_th_star)
        
        if (x_ch < 0):
            Loss = 0
        else:
            Loss = factor1 * (self.W_1**2) * \
                (factor2*x_ch + x_ch**factor3) * self.m_dot_1
        
        return Loss
        
        
    
    







################## PARASITIC LOSSES ###############################################################
###################################################################################################






    
    
    def recirculation_loss(self):
        """
        Function to calculate recirculation enthalpy loss. Formulation is by Whitefield and Baines
        """

        
        D_f = 1 - (self.W_2)/(self.W_1_tip) + ( 0.75*self.Euler_work*self.W_2/self.m_dot_1 )/( ( (((self.N_blades-0.5*self.n_splitter_blades))/math.pi)*( 1 - self.r_1_tip/self.r_2) + (2*self.r_1_tip)/(self.r_2) )*self.W_1_tip*(self.U_2**2) )
        factor = self.loss_params.recirculation
        Loss = factor*((math.tan(abs(self.alpha_2)))**0.5)*((D_f)**2)*(self.U_2**2)*self.m_dot_1
        
        # else:
        #     Loss = 0

        return Loss
    
    
    
    
    
    def leakage_loss(self):
        """
        Function to calculate leakage loss in centrifugal compressor
        """
        factor = self.loss_params.leakage 
        
        r_bar = (self.r_1+self.r_2)/2
        b_bar = (self.b_1+self.b_2)/2
            
        r_shroud = self.r_1_tip+self.s
        
        L_b = (math.pi/4)*(2*self.r_2 - (self.r_1_hub+self.r_1_tip) - self.b_2 + 2*self.L_z )/( math.sin(math.pi/2-abs(self.beta_b1)) + math.sin(math.pi/2-abs(self.beta_b2)) )

        Delta_P_cl = (self.m_dot_1*(self.r_2*self.C_theta_2 - self.r_1*self.C_u_1))/((self.N_blades-0.5*self.n_splitter_blades)*r_bar*b_bar*L_b)
        
        U_cl = 0.816*((2*Delta_P_cl/self.rho_2)**0.5)
        
        m_dot_cl = self.rho_2*(self.N_blades-0.5*self.n_splitter_blades)*self.s*L_b*U_cl
        
        Loss = factor*(m_dot_cl)*(U_cl)*(self.U_2)/(2)

        return Loss

    
    
    
    
    def disk_friction_loss(self):
        """
        Function to calculate the disk friction loss for centrifugal compressor
        """


        factor = self.loss_params.disk_friction
        
        # method 3: DF - 3
        rho_bar = (0.5)*(self.rho_1 + self.rho_2)
        nu_2 = CP.PropsSI('V', 'P', self.P_2, 'T', self.T_2, 'Air')
        Re = (self.U_2*self.r_2)/(nu_2)
        
        if (Re<(3*(10**5))):
            C_f = 3.7*((self.s/self.r_2)**0.1)*(Re**(-0.5))
            
        else:
            C_f = 0.0102*((self.s/self.r_2)**0.1)*(Re**(-0.2))
        
        Loss = factor*(rho_bar*((self.U_2)**3)*((self.r_2)**2)*C_f)/(4)
    
        return Loss
    
    


#################### VANELESS DIFFUSER LOSS ######################################################
##################################################################################################





    def vaneless_diffuser_loss(self, Re_vd):
        """
        Calculate the skin friction coefficient for skin friction loss
        """
        
        factor1 = self.loss_params.vd_factor
        factor2 = self.loss_params.vd_power
        f = factor1*(((1.8*(10**5))/Re_vd)**factor2)

        return f
    










#################### VOLUTE LOSSES ######################################################
##################################################################################################


    def volute_loss(self, C_3, C_r_4, C_theta_4, C_4, r_4, r_5, C_5, A_5, nu):
        

        ########### MERIDIONAL VELOCITY LOSS ###########################################
        if self.volute_meridional_method == 'on':
            meridional_velocity_loss = (C_r_4/C_4)**2

        elif self.volute_meridional_method == 'off':
            meridional_velocity_loss = 0
        else:
            raise ValueError('The volute meridional loss method given is invalid.')
        ###############################################################################

        
        ######### TANGENTIAL VELOCITY LOSS #############################################
        if self.volute_tangential_method == 'on':

            SP = (r_4*C_theta_4)/(r_5 * C_5)
        
            if SP >= 1:
                tangential_velocity_loss = (r_4*(C_theta_4**2)) / (2*r_5*(C_4)**2)*(1-(1/SP**2))
            elif SP < 1:
                tangential_velocity_loss = ((r_4*(C_theta_4)**2) / (r_5*(C_4)**2))*((1-(1/SP))**2)

            tangential_velocity_loss = (1+tangential_velocity_loss)/2

        elif self.volute_tangential_method == 'off':
            tangential_velocity_loss = 0
        
        else:
            raise ValueError('The volute tangential loss method given is invalid.')
        ################################################################################



        #################### COMPUTE SKIN FRICTION LOSS #################################
        if self.volute_skin_friction_method == 'on':

            L_v = math.pi*(r_4 + r_5)/2
            D_h = (4*A_5/math.pi)**0.5

            C_bar = (C_3 + C_4)/2
            D_h = (r_4 - r_5)*2
            Re = C_bar * D_h / nu

            if Re < 2000:
                C_f = 16/Re
            if Re > 2000:
                rel_fric = 1/100
                C_f = (1/4) * (1/(-2 * math.log10(rel_fric/3.71)))**2

            skin_friction_loss = 4*C_f *((C_5/C_4)**2) * (L_v / D_h)

        elif self.volute_skin_friction_method == 'off':
            skin_friction_loss = 0
        else:
            raise ValueError('The volute skin friction loss method given is invalid.')
        
        loss = meridional_velocity_loss + tangential_velocity_loss + skin_friction_loss

        return meridional_velocity_loss, tangential_velocity_loss, skin_friction_loss