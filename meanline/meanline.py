import math
from scipy.optimize import fsolve
import CoolProp.CoolProp as CP
from scipy.integrate import solve_ivp
import numpy as np


from meanline.losses import Centrifugal_Compressor_Losses
from meanline.losses_axial import Axial_Rotor_Losses
from meanline.loss_parameters import LossModelParameters

import warnings

warnings.filterwarnings("ignore")


class MeanLine:
    """
    Class to execute the meanline code procedure
    """
    
    def __init__(self, geometry, loss_params = None):
        
        # fluid properties
        self.C_p = 1005
        self.gamma = 1.4
        self.R = 287

        self.loss_params = loss_params
        self.results = {"performance": {}, "geometry": {}, "losses": {}}

        self.imp_type = geometry['imp_type'] 
        self.P_01 = geometry['P_01']
        self.T_01 = geometry['T_01']
        self.R_tip_1 = geometry['R_tip_1']
        self.R_mean_1 = geometry['R_mean_1']
        self.R_hub_1 = geometry['R_hub_1']
        self.alpha_1 = 0
        self.beta_b1_hub = geometry['beta_b1_hub']
        self.beta_b1_tip = geometry['beta_b1_tip']
        self.beta_b1_mean = geometry['beta_b1_mean']
        self.lambda_1 = geometry['lambda_1']
        self.R_mean_2 = geometry['R_mean_2']
        self.beta_b2 = geometry['beta_b2']
        self.lambda_2 = geometry['lambda_2']
        self.b_2 = geometry['b_2']
        self.L_z = geometry['L_z']
        self.s = geometry['s']
        self.t = geometry['t']
        self.slip_factor = geometry['slip_factor']
        self.n_blades = geometry['nblades']
        self.n_splitter_blades = geometry['n_splitter_blades']
        self.b3 = geometry['b3']
        self.r3 = geometry['r3']


    def degrees_to_radians(self, degrees_value):
        return degrees_value * (math.pi / 180)

    def radians_to_degrees(self, radians_value):
        return radians_value * (180 / math.pi)
    
    def calculate_throat_area(self, R_tip_1, R_hub_1, beta_b1_tip,beta_b1_hub,n_blades,n_splitter_blades,t):
        """
        Function to calculate the throat area
        """

        # alternative method for throat area
        n_values = 10000
        
        delta_r = (R_tip_1 - R_hub_1)/n_values

        delta_beta = ((math.pi/2-abs(beta_b1_tip))-(math.pi/2-abs(beta_b1_hub)))/n_values
    
        summation = 0

        for i in range(n_values):
            r_i = R_hub_1 + (i)*delta_r
            beta_i = (math.pi/2-abs(beta_b1_hub)) + (i)*delta_beta
            summation += (((2*math.pi*r_i)/(n_blades-n_splitter_blades)) - t) * math.sin(beta_i)*delta_r

        A_th = (n_blades-n_splitter_blades)*summation

        return A_th
    



    def calculate_slip_factor(self, slip_factor_model, imp_type, iteration_no, m_dot_1, loss_params):

        if loss_params is None:
            loss_params = LossModelParameters()

        # if only Harrison model is used
        # factor = loss_params.slip_factor_factor
        # power = loss_params.slip_factor_power

        # if flexible Harrison model is used
        factor = loss_params.slip_factor_factor
        power = loss_params.slip_factor_power
        c1 = loss_params.slip_factor_c1
        c2 = loss_params.slip_factor_c2
        c3 = loss_params.slip_factor_c3
        c4 = loss_params.slip_factor_c4

        # Wiesner, Harisson formulation and will also add a modified Harrison formulation for better calibration
        # slip_factor_model = 'Wiesner'
        slip_factor_model = 'Harrison_Flexible'
        # slip_factor_model = 'Harrison'
        
        ################# CALCULATE HARRISON SLIP FACTOR ################
        
        if slip_factor_model == 'Wiesner' and imp_type==1:

            ################ ADOPTER FROM NEW BOOK - BELLINI GAMBINI ####################

            if iteration_no==0:
                self.beta_2 = self.beta_b2

            beta_m = 0.5*(self.beta_1 + self.beta_b2)

            delta_h = self.R_hub_1/self.R_mean_2

            delta_t = ( (delta_h)**2 + (4*self.A_1)/(math.pi*((2*self.R_mean_2)**2)) )**0.5

            zeta = 0.4

            N_blades_slip_factor = int( (2*math.pi*math.cos(abs(beta_m)))/(zeta*math.log(1/delta_t)) )

            slip_factor = 1 - ((math.cos(abs(self.beta_b2)))**0.5)/((N_blades_slip_factor)**0.7)
            
            delta_m = self.R_hub_1 + self.R_tip_1

            Sf_star = math.sin(19*math.pi/180 + 0.2*(90*math.pi/180-self.beta_2))

            delta_m_lim = (slip_factor-Sf_star)/(1-Sf_star)
            
            
            if delta_m>delta_m_lim:
                
                slip_factor = slip_factor*( 1 - ((delta_m-delta_m_lim)/(1-delta_m_lim))**((90*math.pi/180-self.beta_b2)**0.5) )
                
            self.C_theta_2_ideal = slip_factor*self.U_2 - abs(self.C_r_2_ideal * math.tan(abs(self.beta_b2)))
                        
            # print('slip factor:',slip_factor)

        if slip_factor_model=='Harrison' and imp_type==1:
            
            C_u2_PGF = self.U_2 - abs(self.C_r_2_ideal * math.tan(abs(self.beta_b2)))
            
            psi_PGF = (self.U_2*C_u2_PGF-self.U_1*self.C_u1)/(self.U_2**2)
            
            phi_1 = m_dot_1/(self.rho_01*((2*self.R_mean_2)**2)*self.U_2)
            
            phi_2 = (self.C_r_2_ideal)/(self.U_2)
            
            N_s_PGF = (phi_1**0.5)/(psi_PGF**0.75)
            
            T_01 = self.T_1+(self.C_1**2)/(2*self.C_p)
            
            M_u2 = (self.U_2)/((self.gamma*self.R*T_01)**0.5)
            
            ###################### DIFFERENT MODELS ############################

            # ######### GENERAL #######################################
            Ttr = factor*psi_PGF*(M_u2**2)*(((M_u2**2)*phi_1)**power)
            
            slip_factor = ((Ttr)/((self.gamma-1)*(M_u2**2))) + phi_2*math.tan(abs(self.beta_b2))
            
            # print('slip factor:',slip_factor)
            
            self.C_theta_2_ideal = slip_factor*self.U_2 - abs(self.C_r_2_ideal * math.tan(abs(self.beta_b2)))
        
        

        if slip_factor_model=='Harrison_Flexible' and imp_type==1:

            # Calculate base Harrison quantities
            C_u2_PGF = self.U_2 - abs(self.C_r_2_ideal * math.tan(abs(self.beta_b2)))
            
            psi_PGF = (self.U_2*C_u2_PGF - self.U_1*self.C_u1)/(self.U_2**2)
            
            phi_1 = m_dot_1/(self.rho_01*((2*self.R_mean_2)**2)*self.U_2)
            
            phi_2 = (self.C_r_2_ideal)/(self.U_2)
            
            T_01 = self.T_1 + (self.C_1**2)/(2*self.C_p)
            
            M_u2 = (self.U_2)/((self.gamma*self.R*T_01)**0.5)

            
            # Base Harrison TTR
            Ttr = factor * psi_PGF * (M_u2**2) * (((M_u2**2)*phi_1)**power)
            
            # Base Harrison slip factor
            slip_factor_harrison = ((Ttr)/((self.gamma-1)*(M_u2**2))) + phi_2*math.tan(abs(self.beta_b2))
            
            # Polynomial correction based on flow coefficient
            delta_slip = c1 + c2*(phi_1)+c3*(phi_1**2)+c4*(phi_1**3)
            
            # Final slip factor with correction
            slip_factor = slip_factor_harrison + delta_slip
            
            # Calculate tangential velocity component
            self.C_theta_2_ideal = slip_factor*self.U_2 - abs(self.C_r_2_ideal * math.tan(abs(self.beta_b2)))
            
        else:
        
            if (imp_type==0):
                # use slip factor to calculate C_u_2
                self.C_u_2_ideal = self.slip_factor*self.U_2 - abs(self.C_x_2_ideal * math.tan(abs(self.beta_b2)))
                
            if (imp_type==1):
                self.C_theta_2_ideal = self.slip_factor*self.U_2 - abs(self.C_r_2_ideal * math.tan(abs(self.beta_b2)))
        

        # print(f"[SLP]  model={slip_factor_model:8s}  β2={math.degrees(self.beta_b2):5.1f}°  "
        # f"slip={slip_factor:6.3f}")




        
        
    def execution_impeller_inlet(self, m_dot_1, omega, imp_type):
        """
        Execute the calculations in the impeller inlet
        """
        
        # Function to find roots for C_1 (analytically solved simultaneous equations)
        self.A_1 = math.pi*(self.R_tip_1**2 - self.R_hub_1**2)

        # calculate the throat area
        self.A_th = self.calculate_throat_area(self.R_tip_1, self.R_hub_1,self.beta_b1_tip,self.beta_b1_hub,self.n_blades,self.n_splitter_blades,self.t)

        def equation(C_1):
            left_term = ((m_dot_1 * self.R) / (C_1 * self.A_1 * math.cos(self.alpha_1))) * (self.T_01 - (C_1**2) / (2 * self.C_p))
            right_term = (self.T_01 - (C_1**2) / (2 * self.C_p))**(self.gamma / (self.gamma - 1)) * ( self.P_01 / ((self.T_01)**((self.gamma) / (self.gamma-1))))
            return left_term - right_term

        # Initial guess for C_1 (you might need to tweak this if convergence is an issue)
        initial_guess = 1

        # Solve for C_1
        self.C_1 = fsolve(equation, initial_guess)[0]
         
        # based on the value of C_1, calculate rho_1, T_1 and P_1 based on the simultaneous equations
        self.rho_1 = (m_dot_1)/(self.C_1*self.A_1*math.cos(self.alpha_1))
        self.T_1 = self.T_01 - (self.C_1**2)/(2*self.C_p)
        self.P_1 = ((self.T_1)**(self.gamma/(self.gamma-1)))*( self.P_01/((self.T_01)**((self.gamma)/(self.gamma-1))))
        # print('rho1:',self.rho_1,'mdot:',m_dot_1)

        self.rho_01 = CP.PropsSI('D', 'P', self.P_01, 'T', self.T_01, 'Air')  # Density in kg/m^3
       
        # determine the volumetric flow rate
        # self.Q = (m_dot_1*self.T_1*self.R)/(self.P_1)
        self.Q =(m_dot_1)/(self.rho_01)
        # print('volumetric flow rate:',self.Q)
         

        # determine the axial component of flow velocity
        self.C_x1 = self.C_1 *math.cos(self.alpha_1)

        # print('AXIAL VELOCITY:',self.C_x1)

        # determine the speed of sound with static quantities
        self.a_1 = (self.gamma*self.R*self.T_1)**(0.5)

        # calculate the tangential component of flow velocity
        self.C_u1 = self.C_x1 * math.tan(self.alpha_1)

        # calculate the Mach number in the inlet of the impeller
        self.M_1 = self.C_1/self.a_1

        # print(f"[IN ]  C1={self.C_1:8.2f}  rho1={self.rho_1:7.3f}  T1={self.T_1:7.2f}  P1={self.P_1:8.0f}")
        # print('omega:',omega)

        # calculate the relative flow angle at the inlet of the impeller
        
        
        self.U_1 = omega * self.R_mean_1
        self.U_2 = omega * self.R_mean_2
        self.beta_1 = - math.atan( (abs(self.U_1) - abs(self.C_u1))/self.C_x1 )
        self.beta_1_degrees = self.radians_to_degrees(self.beta_1)

        # print('BETA 1 DEGREES:',self.beta_1_degrees)

        if (imp_type=="Centrifugal"):

            # beta_b1 hub and beta_b1_tip are defined by the geometry and we now need to define beta_b1. To do that:
        
            ############## ASSUMING PROPORTIONAL RELATION #################
            # self.beta_b1 = (self.beta_b1_hub)+((self.R_mean_1 - self.R_hub_1)/(self.R_tip_1 - self.R_hub_1))*(self.beta_b1_tip-self.beta_b1_hub)
            # print('beta b1 proportional:',self.beta_b1*-57)

            # Use beta_b1_mean directly from geometry (calculated in preliminary design)
            self.beta_b1 = self.beta_b1_mean
            
            self.beta_b1_degrees = (self.beta_b1)*(180/math.pi)
            
            # calculate the incidence angle in the inlet of the rotor
            self.i_r1 = abs(abs(self.beta_b1) - abs(self.beta_1))

        else:

            self.beta_b1_degrees = (self.beta_b1)*(180/math.pi)
            # calculate the incidence angle in the inlet of the rotor
            self.i_r1 = abs(abs(self.beta_b1) - abs(self.beta_1))
            
        # self.i_r1_degrees = conversion_obj.radians_to_degrees(self.i_r1)

        # print('INCIDENCE ANGLE MEAN:',self.i_r1_degrees, 'MDOT:', m_dot_1, 'SPEED:', omega)

        
        
        beta_1_tip = - math.atan( (abs(omega*self.R_tip_1) - abs(self.C_u1))/self.C_x1 )
        i_r1_tip = abs(abs(self.beta_b1_tip) - abs(beta_1_tip))*(180/math.pi)
        # print('INCIDENCE ANGLE TIP:',i_r1_tip, 'MDOT:', m_dot_1, 'SPEED:', omega)

        beta_1_hub = - math.atan( (abs(omega*self.R_hub_1) - abs(self.C_u1))/self.C_x1 )
        i_r1_hub = abs(abs(self.beta_b1_hub) - abs(beta_1_hub))*(180/math.pi)
        # print('INCIDENCE ANGLE HUB:',i_r1_hub, 'MDOT:', m_dot_1, 'SPEED:', omega)

        # calculate the relative flow velocity in the inlet of the impeller
        self.W_1 = (self.C_x1)/(math.cos(self.beta_1))
        
        # calculate the tangential component of the fluid relative velocity
        self.W_u_1 = self.W_1*math.sin(self.beta_1)
        
        # calculate the relative flow angle at the impeller inlet tip
        self.beta_1_tip = - math.atan( (abs(self.R_tip_1*omega) - abs(self.C_u1))/self.C_x1 )
        self.beta_1_tip_degrees = self.radians_to_degrees(self.beta_1_tip)
        # print('OMEGA:',omega)
        # raise 'END'
        # calculate the relarive velocity at the impeller inlet tip:
        self.W_1_tip = (self.C_x1)/(math.cos(self.beta_1_tip))
        
        # calculate the relative flow angle at the impeller inlet hub
        self.beta_1_hub = - math.atan( (abs(self.R_hub_1*omega) - abs(self.C_u1))/self.C_x1 )
        self.beta_1_hub_degrees = self.radians_to_degrees(self.beta_1_hub)
        
        # calculate the relarive velocity at the impeller inlet tip:
        self.W_1_hub = (self.C_x1)/(math.cos(self.beta_1_hub))
        
        # calculate the relative flow angle at the impeller inlet hub
        self.beta_1_tip = - math.atan( (abs(self.R_tip_1*omega) - abs(self.C_u1))/self.C_x1 )
        self.beta_1_tip_degrees = self.radians_to_degrees(self.beta_1_tip)
        
        # calculate the relative flow velocity at the impeller inlet hub
        self.W_1_hub = (self.C_x1)/(math.cos(self.beta_1_hub))
        
        # calculate the kinematic viscosity at state 01, as it is required by skin friction loss mechanism
        self.mu_01 = CP.PropsSI('V', 'P', self.P_01, 'T', self.T_01, 'Air')  # Dynamic viscosity in Pa.s
        self.nu_01 = self.mu_01/self.rho_01

        
        # calculate the kinematic viscosity at state 1, as it is required by skin friction loss mechanism
        self.mu_1 = CP.PropsSI('V', 'P', self.P_1, 'T', self.T_1, 'Air')  # Dynamic viscosity in Pa.s        
        
        # calculate U_1_tip
        self.U_1_tip = omega*self.R_tip_1
        
        # calculate the relative mach number at impeller inlet
        self.M_w1_tip = self.W_1_tip/((self.gamma*self.R*self.T_1)**0.5)

        # volumetric flow rate in impeller inlet
        V_1 = self.A_1*self.C_x1     

        # print('INCIDENCE ANGLE:',abs(self.beta_2_degrees - 45),'MDOT:',)


            
        
    def execution_impeller_outlet(self,m_dot_1,omega,imp_type):
        """
        Function to execute the calculations in the impeller outlet
        """
        self.A_2 = 2*math.pi*self.R_mean_2*self.b_2

        if (imp_type=="Axial"):
            imp_type = 0
        elif (imp_type=="Centrifugal"):
            imp_type = 1

        # define an initial guess for P2 and the tolerance for the iterative procedure
        P_2_initial = self.P_1*1.2
        error_P2 = 1
        tolerance_P2 = 10**(-5)
        self.P_2 = P_2_initial
        
        iteration_no = 0
        previous_valid_P_2 = P_2_initial  # To store the last valid pressure value

        while (error_P2>tolerance_P2):
            

            # calculate T_2_ideal using isentropic flow relations between states 1 and 2
            self.T_2_ideal = self.T_1 * ((self.P_2/self.P_1)**((self.gamma-1)/self.gamma))
            
            # using equation of state calculate the ideal density at the impeller outlet
            self.rho_2_ideal = (self.P_2)/(self.R*self.T_2_ideal)
            
            # calculate the axial/radial component of ideal flow velocity
            if (imp_type==0):   # if the compressor is axial
                # calculate the C_x_2 using mass flow rate conservation
                self.C_x_2_ideal = (m_dot_1)/(self.rho_2_ideal*self.A_2)
            if (imp_type==1):
                self.C_r_2_ideal = (m_dot_1)/(self.rho_2_ideal*self.A_2)
            
            self.calculate_slip_factor('Harrison',imp_type, iteration_no, m_dot_1, self.loss_params)
           
            # calculate the relative flow velocity in impeller outlet
            if (imp_type==0):   # if the compressor is axial
                self.W_2_ideal = (self.C_x_2_ideal**2 + (abs(self.U_2) - abs(self.C_u_2_ideal))**2)**0.5
            
                #print('W2 ideal:',self.W_2_ideal)            
            if (imp_type==1):   # if the compressor is centrifugal 
                self.W_2_ideal = (self.C_r_2_ideal**2 + (self.U_2 - self.C_theta_2_ideal)**2)**0.5
            
            # calculate the absolute flow velocity
            if (imp_type==0):
                self.C_2_ideal = (self.C_x_2_ideal**2 + self.C_u_2_ideal**2)**0.5
                
            if (imp_type==1):
                self.C_2_ideal = (self.C_r_2_ideal**2 + self.C_theta_2_ideal**2)**0.5
            
            # calculate the total temperature T_02_ideal
            self.T_02_ideal = self.T_2_ideal + (self.C_2_ideal**2)/(2*self.C_p)
            
            # calculate the ideal work done by the compressor
            self.W_ideal = m_dot_1 * self.C_p * (self.T_02_ideal - self.T_01)

            # calculate the relative flow angle at the impeller outlet
            if (imp_type==0):   # if the compressor is axial
                self.beta_2_ideal = - math.atan((abs(self.U_2 - self.C_u_2_ideal))/(self.C_x_2_ideal))
            if (imp_type==1):   # if the compressor is centrifugal
                self.beta_2_ideal = - math.atan((abs(self.U_2 - self.C_theta_2_ideal))/(self.C_r_2_ideal))
            
            # calculate the absolute flow angle at the impeller outlet
            if (imp_type==0):   # if the compressor is axial
                self.alpha_2_ideal = math.atan(self.C_u_2_ideal/self.C_x_2_ideal)
            if (imp_type==1):   # if the compressor is centrifugal
                self.alpha_2_ideal = math.atan(self.C_theta_2_ideal/self.C_r_2_ideal)

            self.mu_2_ideal = CP.PropsSI('V', 'P', self.P_2, 'T', self.T_2_ideal, 'Air')  # Dynamic viscosity in Pa.s
            self.rho_2_ideal = CP.PropsSI('D', 'P', self.P_2, 'T', self.T_2_ideal, 'Air')  # Density in kg/m^3
            self.nu_2_ideal = self.mu_2_ideal/self.rho_2_ideal 
            
            # in the initial iteration, use the ideal values to generate an initial guess for the losses
            if (imp_type==0):    # of the compressor is axial, account for the impeller losses
                if (iteration_no==0):
                    self.loss_obj = Axial_Rotor_Losses(A_1 = self.A_1, A_2 = self.A_2, L_ch = self.B_1, C_u_1 = self.C_u1, C_u_2 = self.C_u_2_ideal, r_1 = self.R_mean_1, C_p = self.C_p, r_2 = self.R_mean_2, r_1_tip = self.R_tip_1, r_1_hub = self.R_hub_1, k_s = self.k_s, T_1 = self.T_1, T_2 = self.T_2_ideal, t = self.t, P_1 = self.P_1, P_2 = self.P_2, W_1 = self.W_1, W_2 = self.W_2_ideal, alpha_1 = self.alpha_1, alpha_2 = self.alpha_2_ideal, beta_1 = self.beta_1, beta_2 = self.beta_2_ideal, gamma = self.gamma, mu_1 =  self.mu_1, rho_1 = self.rho_1, rho_2 = self.rho_2_ideal, sigma = self.sigma, tau = self.s, C_1 = self.C_1, Profile_model_Koch = 0, Profile_model_Konig = 1)
                    
                    # Assessing Pressure Loss Coefficients
                    Y_p = self.loss_obj.Profile_Losses()
                    Y_s = self.loss_obj.Secondary_Losses()
                    Y_ew = self.loss_obj.End_Wall_Losses()
                    Y_shock = self.loss_obj.Shock_Wave_Losses()
                    Y_tc = self.loss_obj.Tip_Clearance_Losses()
                    
                    # Assessing Enthalpy Loss for each coefficent 
                    self.Profile_losses_Rotor = self.loss_obj.Delta_Enthalpy(Y_p) 
                    self.Secondary_losses_Rotor = self.loss_obj.Delta_Enthalpy(Y_s) 
                    self.End_wall_losses_Rotor = self.loss_obj.Delta_Enthalpy(Y_ew) 
                    self.Shock_wave_losses_Rotor = self.loss_obj.Delta_Enthalpy(Y_shock) 
                    self.Tip_clearance_losses_Rotor = self.loss_obj.Delta_Enthalpy(Y_tc)
                    
                    # Total enthalpy loss
                    self.loss = self.Profile_losses_Rotor + self.Secondary_losses_Rotor + self.End_wall_losses_Rotor + self.Shock_wave_losses_Rotor + self.Tip_clearance_losses_Rotor
                    
                else:
                    self.loss_obj = Axial_Rotor_Losses(A_1 = self.A_1, A_2 = self.A_2, L_ch = self.B_1, C_u_1 = self.C_u1, C_u_2 = self.C_u2_real, r_1 = self.R_mean_1, C_p = self.C_p, r_2 = self.R_mean_2, r_1_tip = self.R_tip_1, r_1_hub = self.R_hub_1, k_s = self.k_s, T_1 = self.T_1, T_2 = self.T_2, t = self.t, P_1 = self.P_1, P_2 = self.P_2, W_1 = self.W_1, W_2 = self.W_2_real, alpha_1 = self.alpha_1, alpha_2 = self.alpha_2, beta_1 = self.beta_1, beta_2 = self.beta_2, gamma = self.gamma, mu_1 =  self.mu_1, rho_1 = self.rho_1, rho_2 = self.rho_2, sigma = self.sigma, tau = self.s, C_1 = self.C_1, Profile_model_Koch = 0, Profile_model_Konig = 1)
                    pass
                    #####################################################
                    # Assessing Pressure Loss Coefficients
                    Y_p = self.loss_obj.Profile_Losses()
                    Y_s = self.loss_obj.Secondary_Losses()
                    Y_ew = self.loss_obj.End_Wall_Losses()
                    Y_shock = self.loss_obj.Shock_Wave_Losses()
                    Y_tc = self.loss_obj.Tip_Clearance_Losses()
                    

                    # Assessing Enthalpy Loss for each coefficent 
                    self.Profile_losses_Rotor = self.loss_obj.Delta_Enthalpy(Y_p) 
                    self.Secondary_losses_Rotor = self.loss_obj.Delta_Enthalpy(Y_s) 
                    self.End_wall_losses_Rotor = self.loss_obj.Delta_Enthalpy(Y_ew) 
                    self.Shock_wave_losses_Rotor = self.loss_obj.Delta_Enthalpy(Y_shock) 
                    self.Tip_clearance_losses_Rotor = self.loss_obj.Delta_Enthalpy(Y_tc)
                    
                    # Total enthalpy loss
                    self.loss = self.Profile_losses_Rotor + self.Secondary_losses_Rotor + self.End_wall_losses_Rotor + self.Shock_wave_losses_Rotor + self.Tip_clearance_losses_Rotor

            if (imp_type==1):
                
                if (iteration_no==0):
                    self.loss_obj = Centrifugal_Compressor_Losses(
                        s = self.s,
                        b_2 = self.b_2,
                        r_1_hub = self.R_hub_1,
                        r_1_tip = self.R_tip_1,
                        r_2 = self.R_mean_2,
                        U_2 = self.U_2,
                        C_2 = self.C_2_ideal,
                        nu_01 = self.nu_01,
                        beta_1_tip = self.beta_1_tip,
                        beta_1_hub = self.beta_1_hub,
                        beta_2 = self.beta_2_ideal,
                        N_blades = self.n_blades,
                        L_z = self.L_z,
                        W_2 = self.W_2_ideal,
                        W_1_tip = self.W_1_tip,
                        Euler_work = self.W_ideal,
                        W_u_1 = self.W_u_1,
                        alpha_2 = self.alpha_2_ideal,
                        rho_2 = self.rho_2_ideal,
                        C_theta_2 = self.C_theta_2_ideal,
                        r_1 = self.R_mean_1,
                        C_u_1 = self.C_u1,
                        b_1 =  - self.R_hub_1 + self.R_tip_1,
                        m_dot_1 = m_dot_1,
                        A_2 = self.A_2,
                        gamma = self.gamma,
                        R = self.R,
                        beta_b2 = self.beta_b2,
                        T_2 = self.T_2_ideal,
                        rho_1 = self.rho_1,
                        T_1 = self.T_1,
                        A_1 = self.A_1,
                        beta_b1 = self.beta_b1,
                        t = self.t,
                        rho_01 = self.rho_01,
                        T_01 = self.T_01,
                        U_1 = self.U_1,
                        W_1 = self.W_1,
                        T_02 = self.T_02_ideal,
                        C_1 = self.C_1,
                        nu_2 = self.nu_2_ideal,
                        beta_1 = self.beta_1,
                        W_1_hub = self.W_1_hub,
                        alpha_1 = self.alpha_1,
                        U_1_tip = self.U_1_tip,
                        omega = omega,
                        P_2 = self.P_2,
                        mu_2 = self.mu_2_ideal,
                        beta_b1_tip = self.beta_b1_tip,
                        beta_b1_hub = self.beta_b1_hub,
                        P_1 = self.P_1,
                        C_p = self.C_p,
                        slip_factor = self.slip_factor,
                        n_splitter_blades = self.n_splitter_blades,
                        A_th = self.A_th,
                        loss_params = self.loss_params
                    )
                    
                    self.clearance_loss = self.loss_obj.clearance_loss()
                    self.skin_friction = self.loss_obj.skin_friction_loss()
                    self.blade_loading_loss = self.loss_obj.blade_loading_loss()
                    self.incidence_loss = self.loss_obj.incidence_loss()
                    self.mixing_loss = self.loss_obj.mixing_loss()
                    self.choke_loss = self.loss_obj.choke_loss()
                    self.shock_loss = self.loss_obj.shock_loss()
                    
                    self.recirculation_loss = self.loss_obj.recirculation_loss()
                    self.leakage_loss = self.loss_obj.leakage_loss()
                    self.disk_friction_loss = self.loss_obj.disk_friction_loss()

                    self.loss_int = self.clearance_loss + self.skin_friction + self.blade_loading_loss + self.incidence_loss + self.mixing_loss + self.choke_loss + self.shock_loss
                    self.loss_ext = self.recirculation_loss + self.disk_friction_loss + self.leakage_loss
                    self.loss = self.loss_int + self.loss_ext 
                    # self.loss = 0
                    

                    # print("------ Loss Breakdown ------")
                    # print(f"Clearance Loss:        {self.clearance_loss:.6f}")
                    # print(f"Skin Friction Loss:    {self.skin_friction:.6f}")
                    # print(f"Blade Loading Loss:    {self.blade_loading_loss:.6f}")
                    # print(f"Incidence Loss:        {self.incidence_loss:.6f}")
                    # print(f"Mixing Loss:           {self.mixing_loss:.6f}")
                    # print(f"Choke Loss:            {self.choke_loss:.6f}")
                    # print(f"Shock Loss:            {self.shock_loss:.6f}")
                    # print()
                    # print(f"Recirculation Loss:    {self.recirculation_loss:.6f}")
                    # print(f"Leakage Loss:          {self.leakage_loss:.6f}")
                    # print(f"Disk Friction Loss:    {self.disk_friction_loss:.6f}")
                    # print()
                    # print(f"Internal Loss (sum):   {self.loss_int:.6f}")
                    # print(f"External Loss (sum):   {self.loss_ext:.6f}")
                    # print(f"Total Loss:            {self.loss:.6f}")
                    # print("----------------------------")
                    # raise 'end'

                # otherwise use the real results from the previous iteration
                else:
                    self.loss_obj = Centrifugal_Compressor_Losses(
                        s = self.s,
                        b_2 = self.b_2,
                        r_1_hub = self.R_hub_1,
                        r_1_tip = self.R_tip_1,
                        r_2 = self.R_mean_2,
                        U_2 = self.U_2,
                        C_2 = self.C_2_real,
                        nu_01 = self.nu_01,
                        beta_1_tip = self.beta_1_tip,
                        beta_1_hub = self.beta_1_hub,
                        beta_2 = self.beta_2,
                        N_blades = self.n_blades,
                        L_z = self.L_z,
                        W_2 = self.W_2_real,
                        W_1_tip = self.W_1_tip,
                        Euler_work = self.W_ideal,
                        W_u_1 = self.W_u_1,
                        alpha_2 = self.alpha_2,
                        rho_2 = self.rho_2,
                        C_theta_2 = self.C_theta_2_real,
                        r_1 = self.R_mean_1,
                        C_u_1 = self.C_u1,
                        b_1 = - self.R_hub_1 + self.R_tip_1,
                        m_dot_1 = m_dot_1,
                        A_2 = self.A_2,
                        gamma = self.gamma,
                        R = self.R,
                        beta_b2 = self.beta_b2,
                        T_2 = self.T_2,
                        rho_1 = self.rho_1,
                        T_1 = self.T_1,
                        A_1 = self.A_1,
                        beta_b1 = self.beta_b1,
                        t = self.t,
                        rho_01 = self.rho_01,
                        T_01 = self.T_01,
                        U_1 = self.U_1,
                        W_1 = self.W_1,
                        T_02 = self.T_02,
                        C_1 = self.C_1,
                        nu_2 = self.nu_2,
                        beta_1 = self.beta_1,
                        W_1_hub = self.W_1_hub,
                        alpha_1 = self.alpha_1,
                        U_1_tip = self.U_1_tip,
                        omega = omega,
                        P_2 = self.P_2,
                        mu_2 = self.mu_2,
                        beta_b1_tip = self.beta_b1_tip,
                        beta_b1_hub = self.beta_b1_hub,
                        P_1 = self.P_1,
                        C_p = self.C_p,
                        slip_factor = self.slip_factor,
                        n_splitter_blades = self.n_splitter_blades,
                        A_th = self.A_th,
                        loss_params = self.loss_params
                    )
                    
                    self.clearance_loss = self.loss_obj.clearance_loss()
                    self.skin_friction = self.loss_obj.skin_friction_loss()
                    self.blade_loading_loss = self.loss_obj.blade_loading_loss()
                    self.incidence_loss = self.loss_obj.incidence_loss()
                    self.mixing_loss = self.loss_obj.mixing_loss()
                    self.choke_loss = self.loss_obj.choke_loss()
                    self.shock_loss = self.loss_obj.shock_loss()
                    
                    self.recirculation_loss = self.loss_obj.recirculation_loss()
                    self.leakage_loss = self.loss_obj.leakage_loss()
                    self.disk_friction_loss = self.loss_obj.disk_friction_loss()
                    self.entrance_diffusion_loss = self.loss_obj.entrance_diffusion_loss()


                    self.loss_int = self.clearance_loss+ self.skin_friction+ self.blade_loading_loss+ self.incidence_loss+ self.entrance_diffusion_loss + self.mixing_loss+ self.choke_loss+ self.shock_loss
                    self.loss_ext = self.recirculation_loss + self.disk_friction_loss + self.leakage_loss
                    self.loss = self.loss_int + self.loss_ext 
                    # self.loss = 0
                    
                    # print(
                    #     f"[LOSS] it={iteration_no:02d}  "
                    #     f"clr={self.clearance_loss:8.3f}  skn={self.skin_friction:8.3f}  "
                    #     f"mix={self.mixing_loss:8.3f}  tot={self.loss:8.3f}  kW"
                    # )
                    

                    # print("------ Loss Breakdown ------")
                    # print(f"Clearance Loss:        {self.clearance_loss:.6f}")
                    # print(f"Skin Friction Loss:    {self.skin_friction:.6f}")
                    # print(f"Blade Loading Loss:    {self.blade_loading_loss:.6f}")
                    # print(f"Incidence Loss:        {self.incidence_loss:.6f}")
                    # print(f"Mixing Loss:           {self.mixing_loss:.6f}")
                    # print(f"Choke Loss:            {self.choke_loss:.6f}")
                    # print(f"Shock Loss:            {self.shock_loss:.6f}")
                    # print()
                    # print(f"Recirculation Loss:    {self.recirculation_loss:.6f}")
                    # print(f"Leakage Loss:          {self.leakage_loss:.6f}")
                    # print(f"Disk Friction Loss:    {self.disk_friction_loss:.6f}")
                    # print()
                    # print(f"Internal Loss (sum):   {self.loss_int:.6f}")
                    # print(f"External Loss (sum):   {self.loss_ext:.6f}")
                    # print(f"Total Loss:            {self.loss:.6f}")
                    # print("----------------------------")
                    # raise 'end'

            iteration_no+=1
            
            self.W_real = self.W_ideal + self.loss

            # compute the work coefficient
            self.psi = self.W_real/(m_dot_1*(self.U_2**2))
        
            # compute the flow coefficient
            self.flow_coeff = (self.C_1*math.cos(self.alpha_1))/(self.U_2)

            # calculate T_02_actual from actual work
            self.T_02_real = self.T_01 + (self.W_real)/(m_dot_1*self.C_p)
           
            # guess a value for rho_2
            self.rho_2 = (self.P_2)/(self.R*self.T_02_real)

            error_rho_2 = 1
            tolerance_rho_2 = 10**(-5)

            
            while (error_rho_2>tolerance_rho_2):
            
                # calculate the tangential component of flow velocity actual from work
                if (imp_type==0):   # if the compressor is axial
                    self.C_u2_real = (self.W_real)/(m_dot_1*self.U_2) + (self.U_1*self.C_u1)/(self.U_2)
                if (imp_type==1):   # if the compressor is centrifugal
                    self.C_theta_2_real = (self.W_real)/(m_dot_1*self.U_2) + (self.U_1*self.C_u1)/(self.U_2)

                # calculate the axial/radial component of flow velocity actual from work                    
                if (imp_type==0):   # if the compressor is axial
                    self.C_x2_real = (m_dot_1)/(self.rho_2*self.A_2)
                if (imp_type==1):   # if the compressor is centrifugal
                    self.C_r2_real = (m_dot_1)/(self.rho_2*self.A_2)
                
                # calculate the absolute flow angle in the impeller outlet
                if (imp_type==0):   # if the compressor is axial
                    self.alpha_2 = math.atan(self.C_u2_real/self.C_x2_real)
                if (imp_type==1):   # if the compressor is centrifugal
                    self.alpha_2 = math.atan(self.C_theta_2_real/self.C_r2_real)
                
                # calculate the relative flow angle at the impeller outlet
                if (imp_type==0):   # if the compressor is axial
                    self.beta_2 = - math.atan((abs(self.U_2 - self.C_u2_real))/(self.C_x2_real))
                if (imp_type==1):   # if the compressor is centrifugal
                    self.beta_2 = - math.atan((abs(self.U_2 - self.C_theta_2_real))/(self.C_r2_real))
                 
                # calculate the absolute real flow angle in the impeller outlet
                if (imp_type==0):   # if the compressor is axial
                    self.C_2_real = self.C_u2_real/math.sin(self.alpha_2)
                if (imp_type==1):   # if the compressor is centrifugal
                    self.C_2_real = self.C_theta_2_real/math.sin(self.alpha_2)
                
                # calculate the relatvie flow velocity
                if (imp_type==0):   # if the compressor is axial
                    self.W_2_real = (self.C_x2_real**2 + (self.U_2 - self.C_u2_real)**2)**0.5
                if (imp_type==1):   # if the compressor is centrifugal 
                    self.W_2_real = (self.C_r2_real**2 + (self.U_2 - self.C_theta_2_real)**2)**0.5
                
                # use rothalpy to calculate T_2_new
                self.T_2_new = self.T_1 + (1/(2*self.C_p))*(-self.U_1**2 +  self.U_2**2) + (1/(2*self.C_p))*(self.W_1**2 - self.W_2_real**2)
                
                # calculate rho_2_new
                self.rho_2_new = (self.P_2)/(self.R*self.T_2_new)
                
                # calculate the error in the density in the impeller outlet
                error_rho_2 = abs(self.rho_2_new - self.rho_2)
                
                # updated the static density value in the impeller outlet
                self.rho_2 = 0.1*self.rho_2_new + 0.9*self.rho_2

            # calculate the temperature based on the converged density
            self.T_2 = (self.P_2/(self.R*self.rho_2))
            
            # calculate speed of sound at 2
            self.a_2 = (self.gamma*self.R*self.T_2)**0.5

            # calculate the Mach number at 2
            self.M_2 = self.C_2_real/self.a_2
            
            # calculate the stagnation temperature at the impeller outlet
            self.T_02 = self.T_2*( 1 + ((self.gamma-1)/2)*(self.M_2**2) )
            
            # calculate the angles in 2 in degrees
            self.alpha_2_degrees = self.radians_to_degrees(self.alpha_2)
            self.beta_2_degrees = self.radians_to_degrees(self.beta_2)
            
            # if a loss model is used to calculate the loss, then the efficiency is computed by W_ideal/(W_ideal+W_loss)
            self.eta = self.W_ideal/(self.W_ideal+self.loss)
            
            # new stagnation pressure is computed
            self.P_02_new = self.P_01 * ( self.eta * (self.T_02_real/self.T_01 - 1 ) + 1 )**(self.gamma/(self.gamma-1))
                    
            # new static pressure is computed
            # self.P_2_new = (self.P_02_new) / ( (self.T_02_real/self.T_2)**(self.gamma/(self.gamma-1))  )
            self.P_2_new = (self.P_02_new)/( ( 1 + ((self.gamma-1)/2)*(self.M_2**2) )**((self.gamma)/(self.gamma-1))) 
                
            # the error between the initial value and the final calculated value of static pressure in impeller outlet
            error_P2 = abs(self.P_2_new - self.P_2)
                
            # the value of static pressure 
            self.P_2 = 0.1*self.P_2_new + 0.9*self.P_2

            self.mu_2 = CP.PropsSI('V', 'P', self.P_2, 'T', self.T_2, 'Air')  # Dynamic viscosity in Pa.s
            self.rho_2 = CP.PropsSI('D', 'P', self.P_2, 'T', self.T_2, 'Air')  # Density in kg/m^3
            self.nu_2 = self.mu_2/self.rho_2 
            
            # get performance parameters for impeller only
            self.impeller_eff = ((self.P_02_new/self.P_01)**((self.gamma-1)/(self.gamma))-1)/((self.T_02/self.T_01)-1)
            self.impeller_pressure_ratio = (self.P_02_new/self.P_01)

            if (imp_type==0):
                # calculate the flow coefficient at the impeller outlet
                self.phi_2 = self.C_x2_real/(self.U_2)
            else:
                # calculate the flow coefficient at the impeller outlet
                self.phi_2 = self.C_r2_real/(self.U_2)
            
            iteration_no += 1

            self.diff_factor = 1 - (self.W_2_real/self.W_1) + (( 0.75*((self.U_2*self.C_theta_2_real - self.U_1*self.C_u1)/(self.U_2**2)) )/( (self.W_1/self.mu_2) * ( ((self.n_blades-0.5*self.n_splitter_blades)/math.pi)*(1 - self.R_tip_1/self.R_mean_2) + 2*self.R_tip_1/self.R_mean_2 ) ))
            self.de_haller = self.W_2_real/self.W_1_tip

        # print('de haller', self.de_haller, 'mdot:', m_dot_1, 'speed:', omega)

        # print('FLOW COEFF:',self.phi_2)




        # print('DESIGN PARAMETERS CORRESPONDING')
        # print('mdot:',m_dot_1, 'speed:', omega)
        # print('D_2:',2*self.R_mean_2)
        # print('Hub to tip Ratio:',self.R_hub_1/self.R_mean_2)
        # print('Flow Coefficient:',self.phi_2)
        # print('hehreee22')
        # print('Degree of Reaction:', self.solve_reaction(self.psi, self.phi_2, self.alpha_1, self.R_tip_1/self.R_mean_2))

        
            
            

    def execution_vaneless_diffuser(self,prev_comp_type, prev_comp_subtype, m_dot_1):
        """
        Function to compute the properties in the outlet of a vaneless diffuser
        """

        def update_functions(r, C_m, C_u, rho, T_0, b2, b3, r2, r3):
            """
            Function to compute the different properties at each step of the integration
            """
            gamma = 1.4
            R = 287
            C_p = 1005
            C = (C_m**2 + C_u**2)**0.5
            T = T_0 - (C**2)/(2*C_p)
            alpha = math.atan(C_u/C_m)
            dbdr = (b3-b2)/(r3-r2)
            b = b2 + (dbdr)*(r-r2)
            phi = np.radians(90)
            P = rho*R*T
            return gamma, R, C_p, T, C, alpha, b, phi, dbdr, P


        def calculate_pressure(C_m, C_u, rho, T_0, R, C_p):
            """
            Function to calculate the static pressure at each iteration
            """
            C = (C_m**2 + C_u**2)**0.5
            T = T_0 - (C**2)/(2*C_p)
            P = rho * R * T
            return P


        def calculate_alpha(C_m, C_u):
            """
            Function to calculate the flow angle based on the flow velocity components
            """            
            return np.arctan2(C_u, C_m)  # arctan2 handles division by zero

        
        def calculate_P0(P, C_m, C_u, T_0, gamma, R, C_p):
            """
            Function to calculate the stagnation pressure 
            """
            
            C = (C_m**2 + C_u**2)**0.5
            T = T_0 - (C**2)/(2*C_p)
            M = C / np.sqrt(gamma * R * T)
            
            return P * ((1 + ((gamma - 1) / 2) * M**2)**(gamma / (gamma - 1)))


        def system_of_eqs(r, y, update_functions, T_0, b2, b3, r2, r3):
            """
            Equations of derivatives to be solved at each integration iteration
            """
            
            C_m, C_u, rho = y
            gamma, R, C_p, T, C, alpha, b, phi, dbdr, P = update_functions(r, C_m, C_u, rho, T_0, b2, b3, r2, r3)
            mu = CP.PropsSI('V', 'P', P, 'T', T, 'Air')
            Re_vd = (rho*C*2*(r3-r2))/(mu)
            
            # M_2 = previous_component.C_2_real/((self.gamma*self.R*previous_component.T_2)**0.5)

            f = self.loss_obj.vaneless_diffuser_loss(Re_vd)
            
            dCudr = -C_u/r - (f*(C**2)*math.cos(alpha))/(C_m*b*math.sin(phi))
            dCmdr = ((C_u**2)/r - (f*(C**2)*math.sin(alpha))/(b*math.sin(phi)) + (R*T)/(r) + ((R*T)/(b))*dbdr - (f*(C**2)*math.cos(alpha)*R*C_u)/(C_p*b*C_m*math.sin(phi)) - (R*(C_u**2))/(r*C_p))/(C_m/gamma - (R*T)/(C_m))
            drhodr = (-rho/b)*dbdr - (rho/r) - (rho/C_m)*dCmdr

            return [dCmdr, dCudr, drhodr]


        # Initial conditions at r2 - dependent on the type of the component before the vaneless diffuser
        if (prev_comp_type=="Impeller"):
            r2 = self.R_mean_2
            alpha_2 = self.alpha_2_degrees
            C_2 = self.C_2_real
            P_2 = self.P_2
            T_2 = self.T_2
            P_02 = self.P_02_new
            b2 = self.b_2
            T_0 = self.T_02
            P_01 = self.P_01
            T_01 = self.T_01
            
            if (prev_comp_subtype=="Centrifugal"):
                C_m_at_r2 = self.C_r2_real
                C_u_at_r2 = self.C_theta_2_real
            elif (prev_comp_subtype=="Axial"):
                C_m_at_r2 = self.C_x2_real
                C_u_at_r2 = self.C_u2_real
        elif (prev_comp_type=="Diffuser"):
            r2 = self.r3
            alpha_2 = self.alpha_3
            C_2 = self.C_3
            P_2 = self.P_3
            T_2 = self.T_3
            P_02 = self.P_03
            b2 = self.b3
            T_0 = self.T_03
            C_m_at_r2 = self.C_3*math.cos(self.alpha_3)
            C_u_at_r2 = self.C_3*math.sin(self.alpha_3)
        
        b3 = self.b3
        
        R = self.R
        C_p = self.C_p
        
        rho_at_r2 = (P_2)/(R*T_2)

        initial_conditions = [C_m_at_r2, C_u_at_r2, rho_at_r2]  # Initial values of C_m, C_u, rho at r2

        # Define the range of integration
        r3 = self.r3
        
        r_span = (r2, r3)
        
        self.T_03 = T_0
        
        # Define specific radius points for evaluation
        step_size = 0.0005
        
        r_values = np.arange(r2, r3, step_size)
        if r_values[-1] < r3:
            r_values = np.append(r_values, r3)

        # Ensure the last value in r_values is explicitly set to r3
        r_values[-1] = r3
        
        solution = solve_ivp(system_of_eqs, r_span, initial_conditions, t_eval=r_values, args=(update_functions, T_0, b2, b3, r2, r3))

        # Extracting the solution at r3
        C_m_at_r3, C_u_at_r3, rho_at_r3 = solution.y[:, -1]

        C_m = C_m_at_r3
        C_u = C_u_at_r3
        self.rho_3 = rho_at_r3

        self.C_3 = (C_m**2 + C_u**2)**0.5
        self.T_3 = self.T_03 - (self.C_3**2)/(2*C_p)
        
        self.P_3 = self.rho_3*R*self.T_3
        self.alpha_3 = math.atan(C_u/C_m)
        M_3 = self.C_3/((self.gamma*self.R*self.T_3)**0.5)
        self.P_03 = self.P_3*(( 1 + ((self.gamma-1)/2)*(M_3**2) )**((self.gamma)/(self.gamma-1)))
        
        P_array = [calculate_pressure(C_m, C_u, rho, T_0, R, C_p) for C_m, C_u, rho in zip(solution.y[0], solution.y[1], solution.y[2])]

        C_m_values = solution.y[0]  # C_m for each radius
        C_u_values = solution.y[1]  # C_u for each radius
        rho_values = solution.y[2]  # rho for each radius
        r_values = solution.t  # Radii at which the solution is evaluated

        # Calculations for speed of sound, P, P_0, and Mach numbers
        T_values = T_0 - (C_m_values**2 + C_u_values**2) / (2 * C_p)  # Calculate temperature at each point
        a_values = np.sqrt(self.gamma * self.R * T_values)  # Speed of sound at each point
        C_values = np.sqrt(C_m_values**2 + C_u_values**2)  # Total velocity
        M_values = C_values / a_values  # Mach number
        M_meridional_values = C_m_values / a_values  # Meridional Mach number
        P_values = rho_values * self.R * T_values  # Static pressure
        P_0_values = P_values * (1 + (self.gamma - 1) / 2 * M_values**2) ** (self.gamma / (self.gamma - 1))  # Stagnation pressure
        
        dbdr = (b3-b2)/(r3-r2)

        b_values = []

        for r in r_values:
            b_values.append(b2 + (dbdr)*(r-r2))

        A_values = 2*math.pi*r_values*b_values

        self.pressure_loss = abs(self.P_03 - self.P_02_new)

        ###### FOR STAGE EFFICIENCY ###################################################
            
        # calculate the delta_h
        delta_h = m_dot_1*self.C_p*(self.T_03-self.T_01)
        
        # calculate the diffuser losses
        T_3_s = self.T_2*((self.P_3/self.P_2)**((self.gamma-1)/(self.gamma)))
        self.delta_h_diffuser = m_dot_1*self.C_p*(self.T_3 - T_3_s)
        
        # calculate the efficiency
        self.stage_eff = ((self.P_03/P_01)**((self.gamma-1)/(self.gamma))-1)/((self.T_03/T_01)-1)
        self.pressure_ratio = self.P_03/P_01

        # print('DIFFUSER RECOVERY:', ((self.P_3 - self.P_2)/(self.P_02_new - self.P_2)))

    def calculate_stage_efficiency(self):
        
        return self.stage_eff
    





    def solve_reaction(self, psi, phi, alpha1, delta_t):
        """
        Solves for degree of reaction R using the two coupled equations.
        """

        tan_a1 = np.tan(alpha1)

        # print('FUNCTION INPUTS:','psi:',psi,'alpha 1:',alpha1, 'delta t:', delta_t)

        if abs(alpha1) < 1e-6:
            R_analytical = 1 - psi/2
            return R_analytical


        def xi_from_R(R):
            inside = (1  - (2 * psi / phi**2) * (R + psi/2 - 1) + tan_a1**2 * (1 - delta_t**2) - (2 * psi * delta_t / phi) * tan_a1)
            return np.sqrt(inside)

        def equation(R):
            xi = xi_from_R(R)
            RHS = (1 - psi/2 + (phi**2/(2*psi)) * ((1 - xi**2) + tan_a1**2*(1 - delta_t**2)) - phi * delta_t * tan_a1)
            return R - RHS

        R_initial_guess = 0.1  # typical value
        R_solution = fsolve(equation, R_initial_guess)[0]
        return R_solution
    








    def execution_volute(self,compressor_volute,m_dot_1):
        
        
        self.h_05 = self.h_04
        self.s_05_ideal = self.s_04
        self.s_5_ideal = self.s_05_ideal
        self.P_5 = 1.2*self.P_4

        error_P5 = 1
        tolerance_P5 = 10**(-4)
        iteration_no = 0

        while error_P5 > tolerance_P5:
            

            self.rho_5_ideal = CP.PropsSI('D','P',self.P_5,'S',self.s_5_ideal,self.fluid)
            self.mu_5_ideal = CP.PropsSI('V','P', self.P_5, 'S', self.s_5_ideal, self.fluid)
            self.nu_5_ideal = self.mu_5_ideal / self.rho_5_ideal

            # use definition of mass flow rate to compute C_5_ideal
            self.C_5_ideal = m_dot_1 / (compressor_volute.A5 * self.rho_5_ideal)
            
            # Loss model for volute
            # There are 3 types of volute losses:
            # - meridional velocity head loss
            # - tangential velocity head loss
            # - wall skin friction loss
            if iteration_no == 0:
                self.volute_meridional_loss, self.volute_tangential_loss, self.volute_skin_friction_loss = self.loss_obj.volute_loss(self.C_3, 
                    self.C_r_4_ideal, self.C_theta_4_ideal, self.C_4_ideal, self.r_4, 
                    compressor_volute.r5, self.C_5_ideal, compressor_volute.A5, self.nu_5_ideal)
                
            else:
                self.volute_meridional_loss, self.volute_tangential_loss, self.volute_skin_friction_loss = self.loss_obj.volute_loss(self.C_3, 
                    self.C_r_4, self.C_theta_4, self.C_4, self.r_4, 
                    compressor_volute.r5, self.C_5, compressor_volute.A5, self.nu_5)
            
            # compute overall loss
            self.volute_loss = self.volute_meridional_loss + self.volute_tangential_loss + self.volute_skin_friction_loss
            
            # compute the stagnation pressure in the volute outlet given the pressure loss
            self.P_05 = self.P_04 - (self.P_04 - self.P_4)*self.volute_loss

            self.rho_05 = CP.PropsSI('D','P',self.P_05, 'H',self.h_05, self.fluid)
            self.s_05 = CP.PropsSI('S', 'H', self.h_05, 'P', self.P_05, self.fluid)
            self.s_5 = self.s_05
            self.rho_5 = CP.PropsSI('D', 'S', self.s_5, 'P', self.P_5, self.fluid)
            
            # Calculate the actual velocity
            self.C_5 = m_dot_1 / (self.rho_5 * compressor_volute.A5)
            
            # Update the pressure
            self.h_5 = self.h_05 - self.C_5**2 / 2
            self.P_5_new = CP.PropsSI('P', 'H', self.h_5, 'S', self.s_5, self.fluid)
            error_P5 = abs(self.P_5_new - self.P_5)
            self.P_5 = 0.1*self.P_5_new + 0.9*self.P_5
            self.mu_5 = CP.PropsSI('V','H', self.h_5, 'P', self.P_5, self.fluid)
            self.nu_5 = self.mu_5 / self.rho_5
            iteration_no += 1
        
        self.volute_iteration_no = iteration_no

        # Calculate the remaining state 4 thermodynamic properties
        self.T_5 = CP.PropsSI('T', 'P', self.P_5, 'D', self.rho_5, self.fluid)
        self.a_5 = CP.PropsSI('A', 'P', self.P_5, 'T', self.T_5, self.fluid)
        self.M_5 = self.C_5 / self.a_5
        self.T_05 = CP.PropsSI('T', 'P', self.P_05, 'S', self.s_05, self.fluid)

        # Calculate the overall stage efficiency (impeller + vaneless space + vaned diffuser + volute)
        self.s_05_ideal = self.s_01 # Assume all stages are isentropic (1-2-3-4-5)
        h_05_s_total = CP.PropsSI('H', 'P', self.P_05, 'S', self.s_05_ideal, self.fluid) # This is assume all stages are isentropic (1-2-3)
        self.volute_efficiency = (h_05_s_total - self.h_01)/(self.h_05 - self.h_01)
        self.volute_pressure_ratio = self.P_05/self.P_01