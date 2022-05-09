# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:29:31 2021

@author: jacqu
"""
import numpy as np 
import matplotlib.pyplot as plt
from .constants import *
from .functions import *



class Squeezer : 
    
    def __init__(self , sqz_dB , sqz_phi , loss_inj = 0):
        
        self.sqz_dB  = sqz_dB
        self.sqz_fac = sqz_dB*np.log(10) / 20
        self.sqz_phi = sqz_phi
        self.type = 'Squeezer'
        self.loss = np.sqrt(loss_inj)
        self.tau = np.sqrt(1 - loss_inj)
        
        sqz0 = np.array([[np.exp(self.sqz_fac) , 0 ],[0 , np.exp(-self.sqz_fac)]])
        self.transfer_mat =  self.tau * transfer(sqz0 , rot(self.sqz_phi+np.pi/2)) + self.loss*np.eye(2)


class FilterCavity :

    def __init__(self , L_fc , t1 , t2 = 0 , loss_rt = 0):
        
        self.L_fc = L_fc
        self.type = 'Filter Cavity'
        
        self.t1   = t1 
        self.r1 = np.sqrt(1 - self.t1**2)
        self.t2   = t2 
        self.r2 = np.sqrt(1 - self.t2**2)
        
        self.loss = loss_rt 
        self.tau = np.sqrt(1 - self.loss)
        self.transfer_mat = self.Tmatrix(0 , 0)
        self.delta=None
        
    def r_fc( self , freq , delta ):
        omega = 2*np.pi*freq
        detuning = 2*np.pi*delta
        phi = 2*(omega - detuning)*self.L_fc / c 
        r_rt = np.sqrt( 1 - self.t1**2 - self.loss )
             
        r =  self.r1 - self.t1**2 * r_rt * np.exp(- 1j*phi) / (self.r1 * (1 - r_rt*np.exp(- 1j*phi)))
        
        return r 
    
    def r_fc2(self , freq , delta):
        omega = 2*np.pi*freq
        detuning = 2*np.pi*delta        
        phi = 2*(omega - detuning)*self.L_fc / c 
        epsilon = 2*self.loss/(self.t1**2 + self.loss)
        xsi = 2*phi / (self.t1**2 + self.loss)
        
        return 1 - (2 - epsilon) / (1 + 1j*xsi)
    
    
    def tau_fc(self , freq , delta):
      
        return np.sqrt(1 - (abs(self.r_fc(freq , delta))**2 + abs(self.r_fc(-freq,delta))**2)/2 )
    
    def tau_fc2(self , freq , delta):

        return np.sqrt(1 - (abs(self.r_fc2(freq , delta))**2 + abs(self.r_fc2(-freq,delta))**2)/2 )
    
    
    def Tmatrix(self, freq , delta):
        self.delta = delta
        ref_mat = np.array([[self.r_fc(freq , delta) , 0],
                            [0 , np.conjugate(self.r_fc(-freq, delta))]])
        
        transfer_mat = A2.dot(ref_mat).dot(A2.T) 
        self.transfer_mat = transfer_mat*self.tau + np.eye(2)*self.loss
        
        return transfer_mat

            
class InterferometerLKB : 
    
    def __init__(self , L_ifo , m , Q , freq_m , F  , loss_ifo = 0 ):
        
        self.L_ifo = L_ifo 
        self.m = m 
        self.Q = Q 
        self.freq_m = freq_m
        self.omega_m = 2 * np.pi * self.freq_m

        self.F = F 
        self.loss = np.sqrt(loss_ifo)
        self.tau = np.sqrt(1 - loss_ifo)
        self.type = 'Interferometer'
        

        self.gamma = np.pi / self.F 
        self.t_in = np.sqrt(2*self.gamma)
        self.r_in = np.sqrt(1 - self.t_in**2 )
        self.gamma_m = self.omega_m / self.Q 
        self.roundtrip = 2 * self.L_ifo / c
        self.omega_cav = self.gamma / self.roundtrip 


    def Tmatrix(self, freq , lambda_L , intensity_in):
        
        omega = 2*np.pi*freq
        k = 2 * np.pi / lambda_L
    
        chi = 1 / ( self.m * (self.omega_m**2 - omega**2 - 1j * (self.omega_m * omega / self.Q) ) )
    
        K = ( 16 * self.t_in * intensity_in * h_bar * k**2 / (self.gamma - 1j*omega*self.roundtrip)**2 ) * chi

        K_ter = (32 * h_bar * k**2 * intensity_in * chi) / (self.gamma - 1j * omega * self.roundtrip)**2


        transfer_mat = np.array([[1, 0], [K_ter, 1]])

        self.transfer_mat = transfer_mat*self.tau + np.eye(2)*self.loss

        return  transfer_mat
    

class ReadOut :
    def __init__(self , angle_ro=0 , loss_ro = 0):
        self.loss = np.sqrt(loss_ro)
        self.angle = angle_ro
        self.tau = np.sqrt(1 - loss_ro)
        self.type = 'Read Out'
        self.transfer_mat = self.tau * np.eye(2)
        


class SetUp :
    
    def __init__(self, lambda_L , power ):
        
        #given laser charac 
        self.lambda_L = lambda_L
        self.power    = power
        
        #derived laser charac 
        self.freq_L   = c / self.lambda_L
        self.omega_L  = 2 * np.pi * self.freq_L
        self.k_L = 2 * np.pi / self.lambda_L
        self.intensity_input = self.power / (h_bar * self.omega_L)
        self.SetUp = []
        self.listTmatrices = []
        self.listCoeffs = []
        
        self.cov = np.eye(2)
        
        #ifo 
        self.omega_m = None
        self.m = None
        self.F = None
        self.omega_cav = None
        self.gamma_m = None
        self.Q = None

        #sqz
        
        #fc
        self.delta = None
        
        #ro
        
    def add_Squeezer(self , sqz_dB , sqz_phi , loss_inj) :
        sqz = Squeezer(sqz_dB , sqz_phi , loss_inj)
        self.SetUp.append(sqz)
        
    def add_FilterCavity(self , L_fc , t1 , t2 = 0 , loss_rt=0):
        FC = FilterCavity(L_fc , t1 , t2 , loss_rt)
        self.SetUp.append(FC)
        self.delta = delta
        
    def add_IFO(self , L_ifo , m , Q , freq_m , F , loss_ifo=0):
        ifo = InterferometerLKB(L_ifo , m , Q , freq_m , F , loss_ifo)
        self.F = ifo.F
        self.m = ifo.m
        self.omega_m = ifo.omega_m
        self.Q = ifo.Q
        self.gamma_m = ifo.omega_m/ifo.Q
        self.omega_cav = ifo.omega_cav
        self.SetUp.append(ifo)
        
    def add_ReadOut(self , loss_ro = 0):
        self.SetUp.append(ReadOut(loss_ro))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def variance(self , cov , quad_phi = 0 , dB=True ):
        
        cov_rot = transfer(cov , rot(quad_phi))
        var = np.real(cov_rot[0,0])
        if dB : return 10*np.log10(var)
        else : return var 
        
    def Sxx(self , cov , freq  ,  quad_phi=0 ):
        
        omega = 2*np.pi*freq
        speX = self.lambda_L**2 / (256*self.F**2*self.intensity_input) * self.variance(cov , quad_phi , False) * ( 1 + (omega/self.omega_cav)**2)
        return speX
    
    def get_Spectrum(self , freq_min , freq_max , quad_phi = 0 , Temp=0 ):
    
        setup = self.SetUp
        freq = np.linspace(freq_min , freq_max , int(1e4))
        omega= 2*np.pi*freq 
        speX = []
        vacuum = np.eye(2)
        
        for w in omega : 
            TransferMat = np.eye(2)
            for element in setup : 
                if element.type == 'Squeezer':
                    #print('sqz' , element.transfer_mat)
                    T = element.transfer_mat
                elif element.type == 'Interferometer':
                    #print('ifo' , element.Tmatrix(f , setup.lambda_L , setup.intensity_input))
                    T = element.Tmatrix(w/(2*np.pi) , self.lambda_L , self.intensity_input)
                elif element.type == 'Filter Cavity':
                    #print('fc' , element.Tmatrix(f , 0))
                    T = element.Tmatrix(w/(2*np.pi) , self.delta)

                elif element.type == 'Read Out':
                    T = element.transfer_mat
                    #print('ro' , element.transfer_mat)
                TransferMat = np.matmul(T ,TransferMat)
            cov = transfer(vacuum , TransferMat)
            #print(Transfer)
            Sxx = self.Sxx(cov , w/(2*np.pi) , quad_phi)
            if Temp!=0 and Temp>0:
                speXT =2*self.m*self.gamma_m*k_b*Temp*(abs(Xi(self.m , w , self.omega_m , self.Q))**2)
                speXT2=self.m*self.gamma_m*h_bar*w/np.tanh(h_bar*w/(2*k_b*Temp))
                speX.append(Sxx + speXT)
            elif Temp==0 : speX.append(Sxx)
            elif Temp<0 : print('Negative T ?')

        return speX

    def get_Variance(self, freq_min, freq_max, quad_phi=0 , dB = True):

        setup = self.SetUp
        freq = np.linspace(freq_min, freq_max, int(1e4))
        omega = 2 * np.pi * freq
        speX = []
        vacuum = np.eye(2)

        for w in omega:
            TransferMat = np.eye(2)
            for element in setup:
                if element.type == 'Squeezer':
                    # print('sqz' , element.transfer_mat)
                    T = element.transfer_mat
                elif element.type == 'Interferometer':
                    # print('ifo' , element.Tmatrix(f , setup.lambda_L , setup.intensity_input))
                    T = element.Tmatrix(w / (2 * np.pi), self.lambda_L, self.intensity_input)
                elif element.type == 'Filter Cavity':
                    # print('fc' , element.Tmatrix(f , 0))
                    T = element.Tmatrix(w / (2 * np.pi), self.delta)

                elif element.type == 'Read Out':
                    T = element.transfer_mat
                    # print('ro' , element.transfer_mat)
                TransferMat = np.matmul(T, TransferMat)
            cov = transfer(vacuum, TransferMat)
            # print(Transfer)
            if dB == True : speX.append(self.variance(cov, quad_phi , dB = True))
            elif dB == False : speX.append(self.variance(cov, quad_phi , dB = False))
        return speX
            
            
        
            
            
            
            
                        
                          
                          
                          
                          
                          
                          
                          
                                               