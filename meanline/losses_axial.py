import math as mt
import CoolProp.CoolProp as CP

class Axial_Rotor_Losses:
    
    
    def __init__(self,A_1, A_2 , L_ch ,  C_u_1 , C_u_2, r_1 , C_p, r_2 , r_1_tip , r_1_hub , k_s , T_1 ,T_2,  t , P_1 , P_2 , W_1 , W_2 , alpha_1 , alpha_2 , beta_1 , beta_2 , gamma ,  mu_1 , rho_1 , rho_2, sigma , tau, C_1, Profile_model_Koch , Profile_model_Konig ):
        """
        Initialise parameters needed for Axial Stage compressor loss calculations
        """
        #list of Required Parameters:
        
        self.sigma = sigma      # blade solidity, 1.3 for NASA Rotor 37 
        #Inlet Area 
        self.A_1 = A_1
        #Outlet Area 
        self.A_2 = A_2
        # Chord
        self.L_ch = L_ch
        #Heat capacity at constant Pressure
        self.C_p = C_p
        #Inlet absolute tangential velocity 
        self.C_u_1 = C_u_1
        #Outlet absolute tangential velocity
        self.C_u_2 = C_u_2
        #Inlet mean diameter
        self.D_1 = 2*r_1
        #Outlet mean diameter
        self.D_2 = 2*r_2
        #blade height at inlet
        self.h_b1 = r_1_tip - r_1_hub
        #blade height at outlet
        self.h_b2 = 2*self.A_2/(mt.pi*(self.D_1 + self.D_2))
        #surface roughness
        self.k_s = k_s      # assume value,  20*10**(-6) for NASA Rotor 37
        # Inlet Temperature
        self.T_1 = T_1
        #Outlet Temperature
        self.T_2 = T_2
        #max blade thickness
        self.t = t
        #Inlet pressure
        self.p_1 = P_1
        #Outlet pressure 
        self.p_2 = P_2
        #Inlet relative flow velocity
        self.W_1 = W_1
        #Outlet relative flow velocity
        self.W_2 = W_2
        #Inlet absolute flow angle
        self.alpha_1 = alpha_1
        #Outlet absolute flow angle
        self.alpha_2 = alpha_2
        #Inlet relative flow angle
        self.beta_1 =  beta_1
        #Outlet relative flow angle
        self.beta_2 =  beta_2
        # Inlet Dymanic Viscocity
        self.mu_1 = mu_1
        #self.mu_1 = CP.PropsSI('V', 'T', self.T_1, 'P', self.p_1, 'Air')
        # Ratio of specific heat
        self.gamma = gamma
        #Inlet density
        self.rho_1 = rho_1
        #outlet density 
        self.rho_2 = rho_2
        #tip clearance
        self.tau = tau      # need torethink other symbols for clearance in other sections
        self.C_1 = C_1
        #self.Profile_model = Profile_model  0 if original , 1 if new 
        
        """
        
        For optimising Loss models
        
        """
        self.Profile_model_Koch = Profile_model_Koch 
        self.Profile_model_Konig = Profile_model_Konig

        """
        Calulating the required mach numbers Ma_1 and Ma_x1
        """
        #First calculate the speed of sound
        a_s  = CP.PropsSI('A', 'T', self.T_1, 'P', self.p_1, 'Air')             # speed of sound: a
        a_s_2  = CP.PropsSI('A', 'T', self.T_2, 'P', self.p_2, 'Air')             # speed of sound: a
        
    
        #Mach number is dependant on the relative speed, for a stator this is useless, so use the absolute velocity
        
        Ma_1 = (self.W_1/a_s)
        Ma_2 = (self.W_2/a_s_2)
        
        #meridonial mach number is required for some calculations
        
        Ma_x1 = (self.C_1 * mt.cos(self.alpha_1))/a_s
    
        self.Ma_x1 = Ma_x1
        self.Ma_1 = Ma_1
        self.Ma_2 = Ma_2

        """
        Calculating the Reynolds Number
        """
        #Reynolds numer is also required, though seperate for stator and rotor.
        
        Re_1 = (self.rho_1 * self.W_1 * self.L_ch)/ self.mu_1
    
        
        self.Re_1 = Re_1


        
        """
        Sources for Rotor data:
            
        The Effect of Adding Roughness and Thickness to a Transonic Axial Compressor Rotor
        
        Design and overall performance of four highly loaded, high speed inlet stages for an advanced high-pressure-ratio core compressor
        
        """
    
    
    
    def Pressure_Loss_Coefficient(self):
        """
        Overall Pressure coefficient is sum of individual pressure coefficients
        """

       
        
        Y = self.Profile_Losses() + self.Shock_Wave_Losses() + self.Secondary_Losses() + self.End_Wall_Losses()  + self.Tip_Clearance_Losses() 
        
        value = self.Profile_Losses()
        #print('value:',self.Profile_Losses())
        
        return Y
    

    
    
    def Profile_Losses(self):
        """
        Calculating the Profile Loss Pressure Loss Coefficient 
        """

        if self.Profile_model_Koch == 1:

            if self.D_1 == self.D_2:
                Gamma = ( abs(mt.tan(self.beta_1)) - abs(mt.tan(self.beta_2)))*(mt.cos(self.beta_1)/self.sigma)
            else:    
                Gamma = 2*(( self.D_2 * self.C_u_2) -( self.D_1 * self.C_u_1 ))/( self.sigma * self.W_1 * ( self.D_1 + self.D_2 ))
                
            #Define emprical constants
            K_1 = 0.2445
            K_2 = 0.4458
            K_3 = 0.7688
            K_4 = 0.6024    
            
            
            #Definining the area of the throat
            
            A_throat = self.A_1 - (1/3) * (self.A_1 - self.A_2 ) 
            
            # Define Contraction Ratio
            A_star_throat = ((1 -  K_2 * self.sigma * ( self.t / self.L_ch ))/( mt.cos ( (abs(self.beta_1)  + abs(self.beta_2))/2 )))* (A_throat/self.A_1)
            
            #Find fluid density at the throat 
            
            rho_throat = self.rho_1*( 1 - ( self.Ma_x1**2 / ( 1 - self.Ma_x1**2 ) ) * (1 - A_star_throat - K_1 * self.sigma * Gamma * ( mt.tan(self.beta_1) / mt.cos(self.beta_1) ))) 
            
            # Find equivilent Diffusion equation
            
            Df_eq = ( self.W_1 / self.W_2) * (1 + K_3 * ( self.t / self.L_ch) + K_4 * Gamma) * mt.sqrt(( mt.sin( self.beta_1) - K_1 * self.sigma * Gamma )**2 + ((mt.cos(self.beta_1)*self.rho_1)/(A_star_throat * rho_throat))**2) 
            
            #boundary layer momentum thickness at the blade outlet
            theta_0_2_C = 2.644 * 10**(-3) * Df_eq - 1.519 * 10**(-4) + (6.713 * 10**(-3))/(2.6 - Df_eq)        
            
            #he boundary layer trailing edge shape factor
            if Df_eq <= 2:
                H_0_te = (0.91 + 0.35 * Df_eq ) * (1 + 0.48 * (Df_eq - 1 )**4 + 0.21* (Df_eq - 1)**6)
            else:
                H_0_te = 2.7209
                
            #inlet mach number correction factor 
            n = 2.853 + Df_eq * ( -0.97747 + 0.19477 * Df_eq )
            zeta_M = 1.0 + (0.11757 - (0.16983 * Df_eq))* (self.Ma_1**n) 
            
            #Flow area correction correction factor 
            zeta_H = 0.53 * (self.h_b1 / self.h_b2) + 0.47
            
            # Correction for the Reynolds number 
            # Critical reynolds number
            Re_cr = 100 * self.L_ch/self.k_s
            
            if self.Re_1<= Re_cr :
                if  self.Re_1 >= 2*(10**5): 
                    zeta_Re = ((10**6)/self.Re_1)**0.166
                else:
                    zeta_Re = 1.30626 * ((2*(10**5)/self.Re_1)**0.5)
            else:
                if Re_cr >= 2*10**5:
                    zeta_Re = ((10**6)/Re_cr)**0.166
                else:
                    zeta_Re = 1.30626 * ((2*(10**5)/Re_cr)**0.5)
            
            # Calulating for the shape factor aswell
            
            xi_M = 1 +  ( 1.0725 + Df_eq * (-0.8671 + 0.18043 * Df_eq)) * (self.Ma_1**1.8)
            
            xi_H = 1 + ( ( self.h_b1 / self.h_b2) - 1 ) * (0.0026 * (Df_eq**8) - 0.024)
            
            if self.Re_1 < Re_cr: 
                xi_Re = ((10**6)/self.Re_1)
            else:
                xi_Re = ((10**6)/Re_cr)**0.06
                
            theta_2_c = theta_0_2_C * zeta_H * zeta_M * zeta_Re
            
            H_te = H_0_te * xi_H * xi_M * xi_Re
            
            Y_p = 2 * theta_2_c * (self.sigma/mt.cos(self.beta_2)) * ((mt.cos(self.beta_1)/mt.cos(self.beta_2))**2) *((2 * H_te)/( 3 * H_te - 1)) * ((1 - theta_2_c*( self.sigma *  H_te /mt.cos(self.beta_2)))**(-3))
            
            
            # if (Y_p<0):
            #     print('Gamma:',Gamma)
            #     print('Athroat',A_throat)
            #     print('A star trhouat:',A_star_throat)
            #     print('Df eq',Df_eq)
            #     print('Hte',H_te)
        
        if self.Profile_model_Konig == 1:
            """

            Improved Blade Profile Loss and Deviation Angle Models for Advanced Transonic Compressor Bladings: Part I—A Model for Subsonic Flow
        
            """

            # This analysis Functions only for on design calulations, i.e minimum loss, optimised incidence angle  
            Omega = (self.W_2/mt.sqrt(self.rho_2/self.rho_1))
            Omega = self.A_2 /self.A_1

            Gamma = (2/self.sigma)* ((mt.cos(self.beta_1))**2) * (mt.tan(self.beta_1)- Omega * (self.rho_2/self.rho_1)* mt.tan(self.beta_2))

            if Gamma > 0.2 :
                W_max = self.W_1 * (1.12 + 0.61*Gamma)
            else:
                W_max = self.W_1 *(-1.18*(Gamma**2) + 0.1446*Gamma + 1)


            Df_eq = (1/Omega) * (self.rho_2/self.rho_2) * (mt.cos(self.beta_2)/mt.cos(self.beta_1))* (W_max/self.W_1)
            

            if  1 < Df_eq < 2:
                theta_c = 0.0071 * Df_eq - 0.024
            else:
                theta_c = 0.1786* (Df_eq**2) - 0.7071* Df_eq +0.711

            # For off design calculations
            #
                
            
            #theta_c = theta_c + K *( )
        
            h_2 = 1.08


            Y_p = 2 * (theta_c) * (self.rho_1/self.rho_2) * (self.sigma * (Omega**2)/mt.cos(self.beta_2)) * ((mt.cos(self.beta_1)/mt.cos(self.beta_2))**2) * (((2* h_2)/(3*h_2 - 1))* (1 - theta_c * (self.sigma * h_2/mt.cos(self.beta_2)))**(-3)) * ((1 +((self.gamma - 1)/self.gamma)* (self.Ma_2)**2 ) ** (1/(self.gamma-1)))  
        
            return Y_p
    
    
    
    
    
    
    
    def Secondary_Losses (self):
        
        """
        Calculating the Secondary losses Pressure Loss Coefficient 
        """
        
        #define mean flow angle 
        beta_M = mt.atan((mt.tan(self.beta_1) + mt.tan((self.beta_2)))/2)
        
        #define blade lift coefficient
        c_L = (2/self.sigma) * mt.cos(beta_M) * (mt.tan(self.beta_1) - mt.tan(self.beta_2))
        
        # Calculate secondary losses 
        Y_s = 0.018 * self.sigma * ((mt.cos(self.beta_1)**2/mt.cos(beta_M)**3))*c_L**2
              
        return Y_s
    
    
    
    
    def End_Wall_Losses (self):
        
        """
        Calculating the End wall Pressure Loss Coefficient 
        """
        #Calculate endwall losses
        Y_ew = 0.0146 * (2*self.L_ch/(self.h_b1+self.h_b2))*((mt.cos(self.beta_1)/mt.cos(self.beta_2))**2)
        
        return Y_ew
    
    
    
    def Shock_Wave_Losses (self):
        
        """
        Calculating the Shock wave Pressure Loss Coefficient 
        """
        
        if self.Ma_1 >= 1 :
            Y_shock = 0.32 * self.Ma_1**2 - 0.62 * self.Ma_1 + 0.3
        else: 
            Y_shock = 0
            
        
        
        return Y_shock
    
    
    
    def Tip_Clearance_Losses (self):
        
        """
        Calculating the Tip Clearance Pressure Loss Coefficient 
        """
        
        # define average angle
        beta_M = mt.atan((mt.tan(self.beta_1) + mt.tan((self.beta_2)))/2)
        
        #define blade lift coefficient
        c_L = (2/self.sigma) * mt.cos(beta_M) * (abs(mt.tan(self.beta_1) - mt.tan(self.beta_2)))
        
        #Pressure loss from the tip of the blade
        Y_tip = 1.4 * 0.5 * self.sigma * (2*self.tau/(self.h_b1 + self.h_b2)) * (mt.cos(self.beta_1)**2 / mt.cos(beta_M)**3) * c_L ** 1.5
        
        #Pressure loss from clearanc gap
        Y_gap = 0.0049 * self.sigma * (2*self.L_ch/(self.h_b1 + self.h_b2)) * (mt.sqrt(c_L)/mt.cos(beta_M))
        
        Y_tc = Y_tip + Y_gap
        
        
        return Y_tc
            
        
        
        
        
    def Delta_Enthalpy(self, Y):
        
        """
        Calculating the total loss across the component  
        """
        
        # print('Y:',self.Y)
        Y_value = Y
                
        
        h_1 = CP.PropsSI('H', 'P', self.p_1, 'T', self.T_1 , 'Air')
        
        s_1 = CP.PropsSI( 'S' , 'P' , self.p_1 , 'T' , self.T_1 , 'Air' )
        
        h_2is = h_1 + self.C_p * self.T_1 * ((self.p_2/self.p_1) ** ((self.gamma - 1)/self.gamma) - 1)
        
        h_1tr = h_1 + (self.W_1**2)/2    
        
        p_1tr = CP.PropsSI('P', 'H', h_1tr, 'S', s_1, 'Air')
    
        p_2tr = p_1tr - Y_value * (p_1tr - self.p_1)
    
        h_2tr = h_1tr
    
        s_2 = CP.PropsSI( 'S' , 'P' , p_2tr , 'H' , h_2tr , 'Air' )
    
        h_2 = CP.PropsSI ( 'H' , 'P' , self.p_2 , 'S' , s_2   , 'Air')
    
        delta_h_irr = h_2 - h_2is
        
        return delta_h_irr
    







class Axial_Stator_Losses:
    
    
    def __init__(self,A_2, A_3 , L_ch ,  C_u_2 , C_u_3, r_2 , C_p, r_3 , r_2_tip , r_2_hub , k_s , T_2 ,T_3,  t , P_2 , P_3 , W_2 , W_3 , alpha_2 , alpha_3 , beta_2 , beta_3 , gamma ,  mu_2 , rho_2 , rho_3, sigma , tau, C_2 , C_3, Profile_model_Koch , Profile_model_Konig, Profile_model_Lieblein):
        """
        Initialise parameters needed for Axial Stage compressor loss calculations
        """
        #list of Required Parameters:
        
        self.sigma = sigma      # blade solidity, 1.3 for NASA Rotor 37 
        #Inlet Area 
        self.A_1 = A_2
        #Outlet Area 
        self.A_2 = A_3
        # Chord
        self.L_ch = L_ch
        #Heat capacity at constant Pressure
        self.C_p = C_p
        #Inlet absolute tangential velocity 
        self.C_u_1 = C_u_2
        #Outlet absolute tangential velocity
        self.C_u_2 = C_u_3
        #Inlet mean diameter
        self.D_1 = 2*r_2
        #Outlet mean diameter
        self.D_2 = 2*r_3
        #blade height at inlet
        self.h_b1 = r_2_tip - r_2_hub
        #blade height at outlet
        self.h_b2 = 2*self.A_2/(mt.pi*(self.D_2 + self.D_2))
        #surface roughness
        self.k_s = k_s      # assume value,  20*10**(-6) for NASA Rotor 37
        # Inlet Temperature
        self.T_1 = T_2
        #Outlet Temperature
        self.T_2 = T_3
        #max blade thickness
        self.t = t
        #Inlet pressure
        self.p_1 = P_2
        #Outlet pressure 
        self.p_2 = P_3
        #Inlet relative flow velocity
        self.W_1 = W_2
        #Outlet relative flow velocity
        self.W_2 = W_3
        #Inlet absolute flow angle
        self.alpha_1 = alpha_2
        #Outlet absolute flow angle
        self.alpha_2 = alpha_3
        #Inlet relative flow angle
        self.beta_1 =  abs(beta_2)
        #Outlet relative flow angle
        self.beta_2 =  abs(beta_3)
        # Inlet Dymanic Viscocity
        self.mu_1 = mu_2
        #self.mu_1 = CP.PropsSI('V', 'T', self.T_1, 'P', self.p_1, 'Air')
        # Ratio of specific heat
        self.gamma = gamma
        #Inlet density
        self.rho_1 = rho_2
        #outlet density 
        self.rho_2 = rho_3
        #tip clearance
        self.tau = tau      # need torethink other symbols for clearance in other sections
        self.C_1 = C_2
        self.C_2 = C_3
        #self.Profile_model = Profile_model  0 if original , 1 if new 
        
        """
        
        For optimising Loss models
        
        """
        self.Profile_model_Koch = Profile_model_Koch 
        self.Profile_model_Konig = Profile_model_Konig
        self.Profile_model_Lieblein = Profile_model_Lieblein
        
        """
        Calulating the required mach numbers Ma_1 and Ma_x1
        """
        #First calculate the speed of sound
        a_s_1  = CP.PropsSI('A', 'T', self.T_1, 'P', self.p_1, 'Air')             # speed of sound: a
        a_s_2  = CP.PropsSI('A', 'T', self.T_2, 'P', self.p_2, 'Air')             # speed of sound: a
        
    
        #Mach number is dependant on the Absolute velocity as stator 
        
        self.Ma_1 = self.C_1/a_s_1
        self.Ma_2 = self.C_2/a_s_2
        
        #meridonial mach number is required for some calculations
        
        self.Ma_x1 = (self.C_1 * mt.cos(self.alpha_1))/a_s_1

        


        """
        Calculating the Reynolds Number
        """
        #Reynolds numer is also required, though seperate for stator and rotor.
        self.Re_1 = (self.rho_1 * self.C_1 * self.L_ch )/ self.mu_1
        
    


        
        """
        Sources for Rotor data:
            
        The Effect of Adding Roughness and Thickness to a Transonic Axial Compressor Rotor
        
        Design and overall performance of four highly loaded, high speed inlet stages for an advanced high-pressure-ratio core compressor
        
        """
    
    
    
    def Pressure_Loss_Coefficient(self):
        """
        Overall Pressure coefficient is sum of individual pressure coefficients
        """

       
        
        Y = self.Profile_Losses() + self.Shock_Wave_Losses() + self.Secondary_Losses() + self.End_Wall_Losses()  + self.Tip_Clearance_Losses() 
        
        
        return Y
    

    
    
    def Profile_Losses(self):
        """
        Calculating the Profile Loss Pressure Loss Coefficient 
        """

        if self.Profile_model_Koch == 1:

            if self.D_1 == self.D_2:
                Gamma = ( abs(mt.tan(self.beta_1)) - abs(mt.tan(self.beta_2)))*(mt.cos(self.beta_1)/self.sigma)
            else:    
                Gamma = 2*(( self.D_2 * self.C_u_2) -( self.D_1 * self.C_u_1 ))/( self.sigma * self.W_1 * ( self.D_1 + self.D_2 ))
                
            #Define emprical constants
            K_1 = 0.2445
            K_2 = 0.4458
            K_3 = 0.7688
            K_4 = 0.6024    
            
            
            #Definining the area of the throat
            
            A_throat = self.A_1 - (1/3) * (self.A_1 - self.A_2 ) 
            
            # Define Contraction Ratio
            A_star_throat = ((1 -  K_2 * self.sigma * ( self.t / self.L_ch ))/( mt.cos ( (abs(self.beta_1)  + abs(self.beta_2))/2 )))* (A_throat/self.A_1)
            
            #Find fluid density at the throat 
            
            rho_throat = self.rho_1*( 1 - ( self.Ma_x1**2 / ( 1 - self.Ma_x1**2 ) ) * (1 - A_star_throat - K_1 * self.sigma * Gamma * ( mt.tan(self.beta_1) / mt.cos(self.beta_1) ))) 
            
            # Find equivilent Diffusion equation
            
            Df_eq = ( self.W_1 / self.W_2) * (1 + K_3 * ( self.t / self.L_ch) + K_4 * Gamma) * mt.sqrt(( mt.sin( self.beta_1) - K_1 * self.sigma * Gamma )**2 + ((mt.cos(self.beta_1)*self.rho_1)/(A_star_throat * rho_throat))**2) 
        
            #boundary layer momentum thickness at the blade outlet
            theta_0_2_C = 2.644 * 10**(-3) * Df_eq - 1.519 * 10**(-4) + (6.713 * 10**(-3))/(2.6 - Df_eq)        
            
            #he boundary layer trailing edge shape factor
            if Df_eq <= 2:
                H_0_te = (0.91 + 0.35 * Df_eq ) * (1 + 0.48 * (Df_eq - 1 )**4 + 0.21* (Df_eq - 1)**6)
            else:
                H_0_te = 2.7209
                
            #inlet mach number correction factor 
            n = 2.853 + Df_eq * ( -0.97747 + 0.19477 * Df_eq )
            zeta_M = 1.0 + (0.11757 - (0.16983 * Df_eq))* (self.Ma_1**n) 
            
            #Flow area correction correction factor 
            zeta_H = 0.53 * (self.h_b1 / self.h_b2) + 0.47
            
            # Correction for the Reynolds number 
            # Critical reynolds number
            Re_cr = 100 * self.L_ch/self.k_s
            
            if self.Re_1<= Re_cr :
                if  self.Re_1 >= 2*(10**5): 
                    zeta_Re = ((10**6)/self.Re_1)**0.166
                else:
                    zeta_Re = 1.30626 * ((2*(10**5)/self.Re_1)**0.5)
            else:
                if Re_cr >= 2*10**5:
                    zeta_Re = ((10**6)/Re_cr)**0.166
                else:
                    zeta_Re = 1.30626 * ((2*(10**5)/Re_cr)**0.5)
            
            # Calulating for the shape factor aswell
            
            xi_M = 1 +  ( 1.0725 + Df_eq * (-0.8671 + 0.18043 * Df_eq)) * (self.Ma_1**1.8)
            
            xi_H = 1 + ( ( self.h_b1 / self.h_b2) - 1 ) * (0.0026 * (Df_eq**8) - 0.024)
            
            if self.Re_1 < Re_cr: 
                xi_Re = ((10**6)/self.Re_1)
            else:
                xi_Re = ((10**6)/Re_cr)**0.06
                
            theta_2_c = theta_0_2_C * zeta_H * zeta_M * zeta_Re
            
            H_te = H_0_te * xi_H * xi_M * xi_Re
            
            Y_p = 2 * theta_2_c * (self.sigma/mt.cos(self.beta_2)) * ((mt.cos(self.beta_1)/mt.cos(self.beta_2))**2) *((2 * H_te)/( 3 * H_te - 1)) * ((1 - theta_2_c*( self.sigma *  H_te /mt.cos(self.beta_2)))**(-3))
            
            return Y_p
            
            # if (Y_p<0):
            #     print('Gamma:',Gamma)
            #     print('Athroat',A_throat)
            #     print('A star trhouat:',A_star_throat)
            #     print('Df eq',Df_eq)
            #     print('Hte',H_te)
        
        if self.Profile_model_Konig == 1:
            """

            Improved Blade Profile Loss and Deviation Angle Models for Advanced Transonic Compressor Bladings: Part I—A Model for Subsonic Flow
        
            """

            # This analysis Functions only for on design calulations, i.e minimum loss, optimised incidence angle  
            Omega = (self.W_2/mt.sqrt(self.rho_2/self.rho_1))
            Omega = self.A_2 /self.A_1

            Gamma = (2/self.sigma)* ((mt.cos(self.beta_1))**2) * (abs(mt.tan(self.beta_1))- Omega * (self.rho_2/self.rho_1)* abs(mt.tan(self.beta_2)))

            if Gamma > 0.2 :
                W_ratio = (1.12 + 0.61*Gamma)
            else:
                W_ratio = (-1.18*(Gamma**2) + 0.1446*Gamma + 1)


            Df_eq = (1/Omega) * (self.rho_2/self.rho_1) * (mt.cos(self.beta_2)/mt.cos(self.beta_1))* (W_ratio)
            

            if  1 < Df_eq < 2:
                theta_c = 0.0071 * Df_eq - 0.0029
            else:
                theta_c = 0.1786* (Df_eq**2) - 0.7071* Df_eq +0.711

            # For off design calculations
            #
                
            
            #theta_c = theta_c + K *( )
        
            h_2 = 1.08


            Y_p = 2 * (theta_c) * (self.rho_1/self.rho_2) * (self.sigma * (Omega**2)/mt.cos(self.beta_2)) * ((mt.cos(self.beta_1)/mt.cos(self.beta_2))**2) * (((2* h_2)/(3*h_2 - 1))* (1 - theta_c * (self.sigma * h_2/mt.cos(self.beta_2)))**(-3)) * ((1 +((self.gamma - 1)/self.gamma)* (self.Ma_2)**2 ) ** (1/(self.gamma-1)))  
        
            return Y_p
        
        if self.Profile_model_Lieblein == 1:

            H_ax = 1.08

            Df_eq = (mt.cos(self.beta_2)/mt.cos(self.beta_1))*(1.12 + (0.61/self.sigma)* (mt.cos(self.beta_1)**2)*(mt.tan(self.beta_2)- mt.tan(self.beta_1)))

            theta_c = (0.0045/(1- 0.95* mt.log(Df_eq)))

            Y_p = 2* theta_c *(self.sigma/mt.cos(self.beta_2))* ((mt.cos(self.beta_1)/mt.cos(self.beta_2))**2)

            return Y_p
    
    
    
    
    
    
    
    def Secondary_Losses (self):
        
        """
        Calculating the Secondary losses Pressure Loss Coefficient 
        """
        
        #define mean flow angle 
        beta_M = mt.atan((mt.tan(self.beta_1) + mt.tan((self.beta_2)))/2)
        
        #define blade lift coefficient
        c_L = (2/self.sigma) * mt.cos(beta_M) * (mt.tan(self.beta_1) - mt.tan(self.beta_2))
        
        # Calculate secondary losses 
        Y_s = 0.018 * self.sigma * ((mt.cos(self.beta_1)**2/mt.cos(beta_M)**3))*c_L**2
              
        return Y_s
    
    
    
    
    def End_Wall_Losses (self):
        
        """
        Calculating the End wall Pressure Loss Coefficient 
        """
        #Calculate endwall losses
        Y_ew = 0.0146 * (2*self.L_ch/(self.h_b1+self.h_b2))*(mt.cos(self.beta_1)/mt.cos(self.beta_2))**2
    
        return Y_ew
    
    
    
    def Shock_Wave_Losses (self):
        
        """
        Calculating the Shock wave Pressure Loss Coefficient 
        """
        
        if self.Ma_1 >= 1 :
            Y_shock = 0.32 * self.Ma_1**2 - 0.62 * self.Ma_1 + 0.3
        else: 
            Y_shock = 0
            
        
        
        return Y_shock
    
    
    
    def Tip_Clearance_Losses (self):
        
        """
        Calculating the Tip Clearance Pressure Loss Coefficient 
        """
        
        # define average angle
        beta_M = mt.atan((mt.tan(abs(self.beta_1)) + mt.tan((abs(self.beta_2))))/2)
        
        #define blade lift coefficient
        c_L = (2/self.sigma) * mt.cos(beta_M) * abs((abs(mt.tan(abs(self.beta_1))) - abs(mt.tan(abs(self.beta_2)))))
        
        #Pressure loss from the tip of the blade
        Y_tip = 1.4 * 0.5 * self.sigma * (2*self.tau/(self.h_b1 + self.h_b2)) * (mt.cos(self.beta_1)**2 / mt.cos(beta_M)**3) * c_L ** 1.5
        
        

        #Pressure loss from clearanc gap
        Y_gap = 0.0049 * self.sigma * (2*self.L_ch/(self.h_b1 + self.h_b2)) * (mt.sqrt(c_L)/mt.cos(beta_M))
        
        Y_tc = Y_tip + Y_gap
        
        
        return Y_tc
        
        
            
        
        
        
        
    
        