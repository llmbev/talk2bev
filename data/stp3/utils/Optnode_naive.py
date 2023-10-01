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

def bernstein_coeff_order10_new(n, tmin, tmax, t_actual):
    l = tmax - tmin
    t = (t_actual - tmin) / l

    P0 = scipy.special.binom(n, 0) * ((1 - t) ** (n - 0)) * t ** 0
    P1 = scipy.special.binom(n, 1) * ((1 - t) ** (n - 1)) * t ** 1
    P2 = scipy.special.binom(n, 2) * ((1 - t) ** (n - 2)) * t ** 2
    P3 = scipy.special.binom(n, 3) * ((1 - t) ** (n - 3)) * t ** 3
    P4 = scipy.special.binom(n, 4) * ((1 - t) ** (n - 4)) * t ** 4
    P5 = scipy.special.binom(n, 5) * ((1 - t) ** (n - 5)) * t ** 5
    P6 = scipy.special.binom(n, 6) * ((1 - t) ** (n - 6)) * t ** 6
    P7 = scipy.special.binom(n, 7) * ((1 - t) ** (n - 7)) * t ** 7
    P8 = scipy.special.binom(n, 8) * ((1 - t) ** (n - 8)) * t ** 8
    P9 = scipy.special.binom(n, 9) * ((1 - t) ** (n - 9)) * t ** 9
    P10 = scipy.special.binom(n, 10) * ((1 - t) ** (n - 10)) * t ** 10

    P0dot = -10.0 * (-t + 1) ** 9
    P1dot = -90.0 * t * (-t + 1) ** 8 + 10.0 * (-t + 1) ** 9
    P2dot = -360.0 * t ** 2 * (-t + 1) ** 7 + 90.0 * t * (-t + 1) ** 8
    P3dot = -840.0 * t ** 3 * (-t + 1) ** 6 + 360.0 * t ** 2 * (-t + 1) ** 7
    P4dot = -1260.0 * t ** 4 * (-t + 1) ** 5 + 840.0 * t ** 3 * (-t + 1) ** 6
    P5dot = -1260.0 * t ** 5 * (-t + 1) ** 4 + 1260.0 * t ** 4 * (-t + 1) ** 5
    P6dot = -840.0 * t ** 6 * (-t + 1) ** 3 + 1260.0 * t ** 5 * (-t + 1) ** 4
    P7dot = -360.0 * t ** 7 * (-t + 1) ** 2 + 840.0 * t ** 6 * (-t + 1) ** 3
    P8dot = 45.0 * t ** 8 * (2 * t - 2) + 360.0 * t ** 7 * (-t + 1) ** 2
    P9dot = -10.0 * t ** 9 + 9 * t ** 8 * (-10.0 * t + 10.0)
    P10dot = 10.0 * t ** 9

    P0ddot = 90.0 * (-t + 1) ** 8
    P1ddot = 720.0 * t * (-t + 1) ** 7 - 180.0 * (-t + 1) ** 8
    P2ddot = 2520.0 * t ** 2 * (-t + 1) ** 6 - 1440.0 * t * (-t + 1) ** 7 + 90.0 * (-t + 1) ** 8
    P3ddot = 5040.0 * t ** 3 * (-t + 1) ** 5 - 5040.0 * t ** 2 * (-t + 1) ** 6 + 720.0 * t * (-t + 1) ** 7
    P4ddot = 6300.0 * t ** 4 * (-t + 1) ** 4 - 10080.0 * t ** 3 * (-t + 1) ** 5 + 2520.0 * t ** 2 * (-t + 1) ** 6
    P5ddot = 5040.0 * t ** 5 * (-t + 1) ** 3 - 12600.0 * t ** 4 * (-t + 1) ** 4 + 5040.0 * t ** 3 * (-t + 1) ** 5
    P6ddot = 2520.0 * t ** 6 * (-t + 1) ** 2 - 10080.0 * t ** 5 * (-t + 1) ** 3 + 6300.0 * t ** 4 * (-t + 1) ** 4
    P7ddot = -360.0 * t ** 7 * (2 * t - 2) - 5040.0 * t ** 6 * (-t + 1) ** 2 + 5040.0 * t ** 5 * (-t + 1) ** 3
    P8ddot = 90.0 * t ** 8 + 720.0 * t ** 7 * (2 * t - 2) + 2520.0 * t ** 6 * (-t + 1) ** 2
    P9ddot = -180.0 * t ** 8 + 72 * t ** 7 * (-10.0 * t + 10.0)
    P10ddot = 90.0 * t ** 8
    90.0 * t ** 8

    P = np.hstack((P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10))
    Pdot = np.hstack((P0dot, P1dot, P2dot, P3dot, P4dot, P5dot, P6dot, P7dot, P8dot, P9dot, P10dot)) / l
    Pddot = np.hstack((P0ddot, P1ddot, P2ddot, P3ddot, P4ddot, P5ddot, P6ddot, P7ddot, P8ddot, P9ddot, P10ddot)) / (l ** 2)
    return P, Pdot, Pddot


#####################################################
#####################################################
##################### NEW OPTIMIZER #################
#######################################################
######################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################

class OPTNode_batched():
    def __init__(self, P, Pddot, A_eq, A_obs, Q_smoothness, x_obs, y_obs, num=12, num_obs=4, nvar=11, a_obs=1.0, b_obs=1.0, rho_obs=0.3, rho_eq=10.0, weight_smoothness=10, maxiter=300, eps=1e-7, num_tot=48, batch_size=30):
        super().__init__()
        self.device = 'cpu'
        device = self.device
        self.P = torch.tensor(P, dtype=torch.double).to(device).expand(batch_size, -1, -1)
        self.Pddot = torch.tensor(Pddot, dtype=torch.double).to(device).expand(batch_size, -1, -1)
        self.A_eq = torch.tensor(A_eq, dtype=torch.double).to(device).expand(batch_size, -1, -1)
        self.A_obs = torch.tensor(A_obs, dtype=torch.double).to(device).expand(batch_size, -1, -1)
        self.Q_smoothness = torch.tensor(Q_smoothness, dtype=torch.double).to(device).expand(batch_size, -1, -1)
        self.x_obs = torch.tensor(x_obs, dtype=torch.double).to(device).expand(batch_size, -1, -1)
        self.y_obs = torch.tensor(y_obs, dtype=torch.double).to(device).expand(batch_size, -1, -1)
        
        
        self.num = num
        self.num_obs = num_obs
        self.eps = eps
        self.nvar = nvar        
        self.a_obs = a_obs
        self.b_obs = b_obs        
        self.rho_eq = rho_eq
        self.num_obs = num_obs
        self.maxiter = maxiter
        self.num_tot = num_tot
        self.rho_obs = rho_obs
        self.weight_smoothness = weight_smoothness
        self.batch_size = batch_size        

    def optimize2(self, b, lamda_x, lamda_y):
        device = self.device
        # print(b.shape)
        bx_eq_tensor, by_eq_tensor = torch.split(b, 6, dim=0)
        
        d_obs = torch.ones(self.batch_size, self.num_obs, self.num, dtype=torch.double).to(device)
        alpha_obs = torch.zeros(self.batch_size, self.num_obs, self.num, dtype=torch.double).to(device)
        ones_tensor = torch.ones((self.batch_size, self.num_obs, self.num), dtype=torch.double).to(device)
        # print(self.Pddot.shape)
        # trans = self.Pddot.permute(0, 2, 1)
        # print(trans.shape, " AA")
        # import pdb;pdb.set_trace()
        cost_smoothness = self.weight_smoothness * torch.bmm(self.Pddot.permute(0, 2, 1), self.Pddot)
        cost = cost_smoothness + self.rho_obs * torch.bmm(self.A_obs.permute(0, 2, 1), self.A_obs) + self.rho_eq * torch.bmm(self.A_eq.permute(0, 2, 1), self.A_eq)
        # import pdb;pdb.set_trace()
        for i in range(self.maxiter):
            temp_x_obs = d_obs * torch.cos(alpha_obs) * self.a_obs
            temp_y_obs = d_obs * torch.sin(alpha_obs) * self.b_obs

            # import pdb;pdb.set_trace()
            b_obs_x = self.x_obs.view(self.batch_size, self.num * self.num_obs) + temp_x_obs.view(self.batch_size, self.num * self.num_obs)
            b_obs_y = self.y_obs.view(self.batch_size, self.num * self.num_obs) + temp_y_obs.view(self.batch_size, self.num * self.num_obs)
            # import pdb;pdb.set_trace()
#             print(self.A_eq.permute(0, 2, 1).shape, bx_eq_tensor.unsqueeze(2).shape, " POO", lamda_x.shape)
#             print(self.A_obs.shape, b_obs_x.unsqueeze(2).shape, torch.bmm(self.A_obs.permute(0, 2, 1), b_obs_x.unsqueeze(2)).shape, " obs")
#             print(self.A_eq.shape, bx_eq_tensor.unsqueeze(2).shape, torch.bmm(self.A_eq.permute(0, 2, 1), bx_eq_tensor.unsqueeze(2).permute(1, 0, 2)).shape, " eq")
            
            # print(lamda_x.shape, " lamda_x")
            # import pdb;pdb.set_trace()
            lincost_x = -lamda_x - self.rho_obs * torch.bmm(self.A_obs.permute(0, 2, 1), b_obs_x.unsqueeze(2)) - self.rho_eq * torch.bmm(self.A_eq.permute(0, 2, 1), bx_eq_tensor.unsqueeze(2).permute(1, 0, 2))
            lincost_y = -lamda_y - self.rho_obs * torch.bmm(self.A_obs.permute(0, 2, 1), b_obs_y.unsqueeze(2)) - self.rho_eq * torch.bmm(self.A_eq.permute(0, 2, 1), by_eq_tensor.unsqueeze(2).permute(1, 0, 2))
            # import pdb;pdb.set_trace()
            # lincost_x = lincost_x.view(-1, 1)
            # lincost_y = lincost_y.view(-1, 1)
            # print(cost.shape, " AAA")
            # print(cost_inv.shape)
            # print(cost_inv.shape, lincost_x.shape, lincost_x.T.shape, " lincost")
            cost_inv = torch.zeros_like(cost)
            for j in range(self.batch_size):
                cost_inv[j] = torch.linalg.inv(cost[j])

            sol_x = torch.bmm(-cost_inv, lincost_x)
            sol_y = torch.bmm(-cost_inv, lincost_y)
            # import pdb;pdb.set_trace()
            
            # print(torch.linalg.lstsq(lincost_x, -cost).solution)
            # sol_x, _ = torch.linalg.lstsq(lincost_x, -cost)
            # sol_x = torch.linalg.lstsq(lincost_x, -cost).solution
            # sol_y, _ = torch.linalg.lstsq(lincost_y, -cost)
            # sol_y = torch.linalg.lstsq(lincost_y, -cost).solution
            # print(sol_x.shape, self.P.shape, " sol")

            # sol_x = sol_x.view(-1)
            # sol_y = sol_y.view(-1)

            x = torch.bmm(self.P, sol_x)
            y = torch.bmm(self.P, sol_y)
            # import pdb;pdb.set_trace()

            # print(x.shape, y.shape, " xy")
            # print(self.x_obs.shape, " x_obs")
            wc_alpha = x.permute(0, 2, 1) - self.x_obs
            ws_alpha = y.permute(0, 2, 1) - self.y_obs
            alpha_obs = torch.atan2(ws_alpha * self.a_obs, wc_alpha * self.b_obs)
            # import pdb;pdb.set_trace()

            c1_d = self.rho_obs * (self.a_obs ** 2 * torch.cos(alpha_obs) ** 2 + self.b_obs ** 2 * torch.sin(alpha_obs) ** 2)
            c2_d = self.rho_obs * (self.a_obs * wc_alpha * torch.cos(alpha_obs) + self.b_obs * ws_alpha * torch.sin(alpha_obs))
            d_temp = c2_d / c1_d
            d_obs = torch.max(d_temp, ones_tensor)
            # import pdb;pdb.set_trace()

            # print(alpha_obs.shape, wc_alpha.shape, " alpga_obs")
            res_x_obs_vec = wc_alpha - self.a_obs * d_obs * torch.cos(alpha_obs)
            res_y_obs_vec = ws_alpha - self.b_obs * d_obs * torch.sin(alpha_obs)
            # import pdb;pdb.set_trace()

            # print(torch.bmm(self.A_eq, sol_x).shape, " res_eq_x_vec")
            res_eq_x_vec = torch.bmm(self.A_eq, sol_x) - bx_eq_tensor.unsqueeze(2).permute(1, 0, 2)
            res_eq_y_vec = torch.bmm(self.A_eq, sol_y) - by_eq_tensor.unsqueeze(2).permute(1, 0, 2)
            # import pdb;pdb.set_trace()

            # print(res_x_obs_vec.shape, "res_x_obs_vec")
            # print(res_eq_x_vec.shape, "res_eq_x_vec")
            # print(self.A_obs.permute(0, 2, 1).shape, res_x_obs_vec.view(self.batch_size, -1).unsqueeze(2).shape)
            # print(self.A_eq.permute(0, 2, 1).shape, res_eq_x_vec.shape)
            # print(torch.bmm(self.A_obs.permute(0, 2, 1), res_x_obs_vec.view(self.batch_size, -1).unsqueeze(2)).shape)
            # print(torch.bmm(self.A_eq.permute(0, 2, 1), res_eq_x_vec).shape)
            # print(lamda_x.shape)
            lamda_x -= self.rho_obs * torch.bmm(self.A_obs.permute(0, 2, 1), res_x_obs_vec.view(self.batch_size, -1).unsqueeze(2)) + self.rho_eq * torch.bmm(self.A_eq.permute(0, 2, 1), res_eq_x_vec)
            lamda_y -= self.rho_obs * torch.bmm(self.A_obs.permute(0, 2, 1), res_y_obs_vec.view(self.batch_size, -1).unsqueeze(2)) + self.rho_eq * torch.bmm(self.A_eq.permute(0, 2, 1), res_eq_y_vec)
            # import pdb;pdb.set_trace()

        sol = torch.cat([sol_x, sol_y], dim=1)
        return sol
    
    
    def solve(self, b, lamda_x, lamda_y):
        device = self.device
        batch_size, _ = b.size()
        b = b.transpose(0, 1)
        lamda_x = lamda_x.unsqueeze(2)
        lamda_y = lamda_y.unsqueeze(2)
        sol = self.optimize2(b, lamda_x, lamda_y)
        return sol.squeeze(), None