import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import scipy.special
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.linalg import block_diag
from torch.utils.data import Dataset, DataLoader

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import scipy.special
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.linalg import block_diag
from torch.utils.data import Dataset, DataLoader
#from bernstein import bernstesin_coeff_order10_new

class OPTNode_waypoint():
    def __init__(self, rho_eq=1.0, rho_goal=1.0, rho_nonhol=1.0, rho_psi=1.0, maxiter=5000, weight_smoothness=1.0, weight_smoothness_psi=1.0, t_fin=2.0, num=30, bernstein_coeff_order10_new=None, device="cpu"):
        super().__init__()
        self.rho_eq = rho_eq
        self.rho_goal = rho_goal
        self.rho_nonhol = rho_nonhol
        self.rho_psi = rho_psi
        self.maxiter = maxiter
        self.weight_smoothness = weight_smoothness
        self.weight_smoothness_psi = weight_smoothness_psi

        self.device = device        
        
        self.t_fin = t_fin
        self.num = num
        self.t = self.t_fin / self.num

        #self.num_batch = 10
        
        tot_time = np.linspace(0.0, self.t_fin, self.num)
        tot_time_copy = tot_time.reshape(self.num, 1)
        self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
        self.nvar = np.shape(self.P)[1]
        
        self.cost_smoothness = self.weight_smoothness * np.dot(self.Pddot.T, self.Pddot)
        self.cost_smoothness_psi = self.weight_smoothness_psi * np.dot(self.Pddot.T, self.Pddot)
        self.lincost_smoothness_psi = np.zeros(self.nvar)

        # self.A_eq = np.vstack((self.P[0], self.P[9], self.P[19], self.P[-1]))
        # self.A_eq_psi = np.vstack((self.P[0], self.Pdot[0], self.P[-1]))
        self.A_eq = np.vstack((self.P[0], self.P[5], self.P[10], self.P[15], self.P[20], self.P[25], self.P[30]))
        # self.A_eq = np.vstack((self.P[0], self.P[5], self.P[10], self.P[15], self.P[20], self.P[25], self.P[30]))s
        self.A_eq_psi = np.vstack((self.P[0], self.Pdot[0], self.P[-1]))
        
        
        self.P = torch.tensor(self.P, dtype=torch.double).to(device)
        self.Pdot = torch.tensor(self.Pdot, dtype=torch.double).to(device)
        self.Pddot = torch.tensor(self.Pddot, dtype=torch.double).to(device)
        self.A_eq = torch.tensor(self.A_eq, dtype=torch.double).to(device)        
        self.A_eq_psi = torch.tensor(self.A_eq_psi, dtype=torch.double).to(device)
        self.cost_smoothness = torch.tensor(self.cost_smoothness, dtype=torch.double).to(device)
        self.cost_smoothness_psi = torch.tensor(self.cost_smoothness_psi, dtype=torch.double).to(device)
        self.lincost_smoothness_psi = torch.tensor(self.lincost_smoothness_psi, dtype=torch.double).to(device)
        
        self.A_nonhol = self.Pdot
        self.A_psi = self.P
        
        self.lamda_x = None
        self.lamda_y = None
        self.lamda_psi = None
        
    def compute_x(self, v, psi, b_eq_x, b_eq_y):
        b_nonhol_x = v * torch.cos(psi)
        b_nonhol_y = v * torch.sin(psi)
    
        cost = self.cost_smoothness + self.rho_nonhol * torch.matmul(self.A_nonhol.T, self.A_nonhol) + self.rho_eq * torch.matmul(self.A_eq.T, self.A_eq)
        lincost_x = -self.lamda_x - self.rho_nonhol * torch.matmul(self.A_nonhol.T, b_nonhol_x.T).T - self.rho_eq * torch.matmul(self.A_eq.T, b_eq_x.T).T
        lincost_y = -self.lamda_y - self.rho_nonhol * torch.matmul(self.A_nonhol.T, b_nonhol_y.T).T - self.rho_eq * torch.matmul(self.A_eq.T, b_eq_y.T).T

        cost_inv = torch.linalg.inv(cost)

        sol_x = torch.matmul(-cost_inv, lincost_x.T).T
        sol_y = torch.matmul(-cost_inv, lincost_y.T).T

        x = torch.matmul(self.P, sol_x.T).T
        xdot = torch.matmul(self.Pdot, sol_x.T).T

        y = torch.matmul(self.P, sol_y.T).T
        ydot = torch.matmul(self.Pdot, sol_y.T).T
         
        return sol_x, sol_y, x, xdot, y, ydot
    
    def compute_psi(self, psi, lamda_psi, psi_temp, b_eq_psi):
        cost = self.cost_smoothness_psi + self.rho_psi * torch.matmul(self.A_psi.T, self.A_psi) + self.rho_eq * torch.matmul(self.A_eq_psi.T, self.A_eq_psi)
        lincost_psi = -self.lamda_psi - self.rho_psi * torch.matmul(self.A_psi.T, psi_temp.T).T - self.rho_eq * torch.matmul(self.A_eq_psi.T, b_eq_psi.T).T

        cost_inv = torch.linalg.inv(cost)

        sol_psi = torch.matmul(-cost_inv, lincost_psi.T).T

        psi = torch.matmul(self.P, sol_psi.T).T

        res_psi = torch.matmul(self.A_psi, sol_psi.T).T - psi_temp
        res_eq_psi = torch.matmul(self.A_eq_psi, sol_psi.T).T - b_eq_psi

        self.lamda_psi = self.lamda_psi - self.rho_psi * torch.matmul(self.A_psi.T, res_psi.T).T - self.rho_eq * torch.matmul(self.A_eq_psi.T, res_eq_psi.T).T

        return sol_psi, np.linalg.norm(res_psi), np.linalg.norm(res_eq_psi), psi

    
    def solve(self, fixed_params, variable_params):
        batch_size, _ = fixed_params.size()
        x_init, y_init, v_init, psi_init, psidot_init = torch.chunk(fixed_params, 5, dim=1)
        x_fin, y_fin, psi_fin, x_mid1, y_mid1, x_mid2, y_mid2, x_mid3, y_mid3, x_mid4, y_mid4, x_mid5, y_mid5 = torch.chunk(variable_params, 13, dim=1)

        b_eq_x = torch.cat((x_init, x_mid1, x_mid2, x_mid3, x_mid4, x_mid5, x_fin), dim=1)
        b_eq_y = torch.cat((y_init, y_mid1, y_mid2, y_mid3, y_mid4, y_mid5, y_fin), dim=1)
        
        b_eq_psi = torch.cat((psi_init, psidot_init, psi_fin), dim=1)
        
        v = torch.ones(batch_size, self.num, dtype=torch.double).to(self.device) * v_init
        psi = torch.ones(batch_size, self.num, dtype=torch.double).to(self.device) * psi_init
        xdot = v * torch.cos(psi)
        ydot = v * torch.sin(psi)
        
        self.lamda_x = torch.zeros(batch_size, self.nvar, dtype=torch.double).to(self.device)
        self.lamda_y = torch.zeros(batch_size, self.nvar, dtype=torch.double).to(self.device)
        self.lamda_psi = torch.zeros(batch_size, self.nvar, dtype=torch.double).to(self.device)
        
        res_psi_arr = []
        res_eq_psi_arr = []
        res_eq_arr = []
        res_nonhol_arr = []
        for i in range(0, self.maxiter):
            psi_temp = torch.atan2(ydot, xdot)
            c_psi, res_psi, res_eq_psi, psi = self.compute_psi(psi, self.lamda_psi, psi_temp, b_eq_psi)
            c_x, c_y, x, xdot, y, ydot = self.compute_x(v, psi, b_eq_x, b_eq_y)
            
            res_eq_psi_arr.append(res_eq_psi)
            res_psi_arr.append(res_psi)
            v = torch.sqrt(xdot ** 2 + ydot ** 2)
            #v[:, 0] = v_init[:, 0]

            res_eq_x = torch.matmul(self.A_eq, c_x.T).T - b_eq_x
            res_nonhol_x = xdot - v * torch.cos(psi)

            res_eq_y = torch.matmul(self.A_eq, c_y.T).T - b_eq_y
            res_nonhol_y = ydot - v * torch.sin(psi)

            res_eq_arr.append(np.linalg.norm(np.sqrt(res_eq_x**2 + res_eq_y**2)))
            res_nonhol_arr.append(np.linalg.norm(np.sqrt(res_nonhol_x**2 + res_nonhol_y**2)))

            self.lamda_x = self.lamda_x - self.rho_eq * torch.matmul(self.A_eq.T, res_eq_x.T).T - self.rho_nonhol * torch.matmul(self.A_nonhol.T, res_nonhol_x.T).T
            self.lamda_y = self.lamda_y - self.rho_eq * torch.matmul(self.A_eq.T, res_eq_y.T).T - self.rho_nonhol * torch.matmul(self.A_nonhol.T, res_nonhol_y.T).T
        
        primal_sol = torch.hstack((c_x, c_y, c_psi, v))
        return primal_sol, None
    
    def objective(self, fixed_params, variable_params, y):
        c_x = y[:, :self.nvar]
        c_y = y[:, self.nvar:2*self.nvar]
        c_psi = y[:, 2*self.nvar:3*self.nvar]
        v = y[:, 3*self.nvar:]
        
        x_init, y_init, v_init, psi_init, psidot_init = torch.chunk(fixed_params, 5, dim=1)
        x_fin, y_fin, psi_fin, x_mid1, y_mid1, x_mid2, y_mid2, x_mid3, y_mid3, x_mid4, y_mid4, x_mid5, y_mid5 = torch.chunk(variable_params, 13, dim=1)
        
        x = torch.matmul(self.P, c_x.T).T
        y = torch.matmul(self.P, c_y.T).T
        psi = torch.matmul(self.P, c_psi.T).T
        xdot = torch.matmul(self.Pdot, c_x.T).T
        ydot = torch.matmul(self.Pdot, c_y.T).T
        psidot = torch.matmul(self.Pdot, c_psi.T).T
        xddot = torch.matmul(self.Pddot, c_x.T).T
        yddot = torch.matmul(self.Pddot, c_y.T).T
        psiddot = torch.matmul(self.Pddot, c_psi.T).T
        
        cost_nonhol = 0.5*self.rho_nonhol*torch.sum((xdot - v*torch.cos(psi)) ** 2, 1) + 0.5*self.rho_nonhol*torch.sum((ydot - v*torch.sin(psi)) ** 2, 1)
        cost_pos = 0.5*self.rho_eq*(torch.sum((x[:, -1] - x_fin) ** 2, 1) + torch.sum((y[:, -1] - y_fin) ** 2, 1) + torch.sum((x[:, 0] - x_init) ** 2, 1) + torch.sum((y[:, 0] - y_init) ** 2, 1))
        cost_pos += 0.5*self.rho_eq*(torch.sum((x[:, 5] - x_mid1)**2, 1) 
                                     + torch.sum((y[:, 5] - y_mid1)**2, 1) 
                                     + torch.sum((x[:, 10] - x_mid2)**2, 1) 
                                     + torch.sum((y[:, 10] - y_mid2)**2, 1)
                                     + torch.sum((x[:, 15] - x_mid3)**2, 1)
                                     + torch.sum((y[:, 15] - y_mid3)**2, 1)
                                     + torch.sum((x[:, 20] - x_mid4)**2, 1)
                                     + torch.sum((y[:, 20] - y_mid4)**2, 1) 
                                     + torch.sum((x[:, 25] - x_mid5)**2, 1) 
                                     + torch.sum((y[:, 25] - y_mid5)**2, 1)
                                    )
        cost_psi = 0.5*self.rho_eq*(torch.sum((psi[:, -1] - psi_fin) ** 2, 1) + torch.sum((psi[:, 0] - psi_init) ** 2, 1)
                                    + torch.sum((psidot[:, 0] - psidot_init) ** 2, 1))
        #cost_v = 0.5*self.rho_eq*torch.sum((v[:, 0] - v_init) ** 2, 1)
        cost_cancel = torch.diagonal(torch.matmul(-self.lamda_x, c_x.T) + torch.matmul(-self.lamda_y, c_y.T) + torch.matmul(-self.lamda_psi, c_psi.T))
        
        cost_smoothness = 0.5*self.weight_smoothness*(torch.sum(xddot**2, 1) + torch.sum(yddot**2, 1)) + 0.5*self.weight_smoothness_psi*torch.sum(psiddot**2, 1)
        return cost_nonhol + cost_pos + cost_psi + cost_smoothness + cost_cancel #+ cost_v 
