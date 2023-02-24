import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import ot
from ot.gromov import gromov_wasserstein2


import numpy as np
from helper import *
from siren_pytorch import *


'''
Learning with SIREN using proximal gradient
'''
class IGNR(nn.Module):
    '''
    - d_hidden: hidden units for the MLP of SIREN; [h1,h2] means 2 layers, h1 units in first layer, h2 units in second layer
    - w0: frequency for SINE activation
    '''

    def __init__(self,d_hidden=[20,20,20],w0=30.,device='cpu'):
        super(IGNR, self).__init__()
        self.net = SirenNet(
                            dim_in = 2, # input [x,y] coordinate
                            dim_hidden = d_hidden,
                            dim_out = 1, # output graphon (edge) probability 
                            num_layers = len(d_hidden), # f_theta number of layers
                            final_activation = 'sigmoid',
                            w0_initial = w0)
        self.device = device
    def sample(self,M,f_sample='fixed'):
        if f_sample=='fixed':
            x = (torch.arange(M)+(1/2))/M
            y = (torch.arange(M)+(1/2))/M
        else:
            x = torch.sort(torch.rand(M))[0]
            y = x
        xx,yy = torch.meshgrid(x,y)
        mgrid=torch.stack([xx,yy],dim=-1)
        mgrid=rearrange(mgrid, 'h w c -> (h w) c')  
        C_recon_tmp = self.net(mgrid.to(self.device))
        C_recon_tmp = torch.squeeze(rearrange(C_recon_tmp, '(h w) c -> h w c', h = M, w = M))

        # when training only half plane
        C_recon_tmp = torch.triu(C_recon_tmp,diagonal=1)
        C_recon_tmp = C_recon_tmp+torch.transpose(C_recon_tmp,0,1)
        return C_recon_tmp

    def fun_loss_pg(self,M,h_recon,C_input,h_input,f_sample='fixed',G0_prior=None,G0_cost=None):
        # loss that compute gw2 using proximal gradient from Xu et al 2021
        C_recon_tmp = self.sample(M,f_sample=f_sample)
        loss,T = gwloss_pg(C_recon_tmp,C_input,h_recon,h_input,G0_prior=G0_prior,G0_cost=G0_cost)
        return loss,T

    def fun_loss_cg(self,M,h_recon,C_input,h_input,f_sample='fixed'):
        # loss that compute gw2 using the default conditional gradient method 
        C_recon_tmp = self.sample(M,f_sample=f_sample)
        loss,log = gromov_wasserstein2(C_recon_tmp,C_input,h_recon,h_input,log=True)
        return loss,log['T']



class IGNR_pg_wrapper:
    '''
    Traing IGNR_pg wrapper
    '''

    def __init__(self,d_hidden=[10,10,10],w0=30.):
        self.mlp = IGNR(d_hidden,w0)
    def train(self, graphs, K='input', n_epoch=10,lr=0.01,f_sample='fixed'):
        optim   = torch.optim.Adam([*self.mlp.parameters()],lr=lr)

        M = len(graphs)
        loss_l=[]


        trans = [None]*M

        for epoch in range(n_epoch):

            loss = []

            for i in range(M):
                num_node_i = graphs[i].shape[0]
                h_input = torch.from_numpy(ot.unif(num_node_i))
                g_input = torch.from_numpy(graphs[i]).to(torch.float32)

                if K=='input':
                    # reconstruct each same size as input
                    h_recon = torch.from_numpy(ot.unif(num_node_i))
                    K_recon = num_node_i
                else:
                    # reconstruct to a specified size K
                    h_recon = torch.from_numpy(ot.unif(K))
                    K_recon = K

                loss_i,T_i = self.mlp.fun_loss_pg(K_recon,h_recon,g_input,h_input,f_sample=f_sample,G0_prior=None,G0_cost=trans[i]) 
                loss.append(loss_i)
                trans[i]=T_i

            loss = torch.stack(loss)
            loss = torch.mean(loss)
            loss.backward()
            optim.step()
            optim.zero_grad()
            loss_l.append(loss.item())
            #print(str(epoch)+':'+str(loss.item()))

        return loss_l

    def get_W(self,K):
        W = self.mlp.sample(K)
        return W.detach().cpu().numpy()



class IGNR_cg_wrapper:
    '''
    - reconstruction size K can be of varying size: same size as each input graph
    - reconstruction size can also be of 
    - this version just use uniform weigths for computation (no pre-computation such as normalized deg)
    

    '''

    def __init__(self,d_hidden=[10,10,10],w0=30.):
        self.mlp = IGNR(d_hidden,w0)
    def train(self, aligned_graphs, K='input',n_epoch=10,lr=0.01,f_sample='fixed'):
        optim   = torch.optim.Adam([*self.mlp.parameters()],lr=lr)

        M = len(aligned_graphs)
        loss_l=[]

        for epoch in range(n_epoch):

            loss = []

            for i in range(M):
                num_node_i = aligned_graphs[i].shape[0]
                h_input = torch.from_numpy(ot.unif(num_node_i))
                g_input = torch.from_numpy(aligned_graphs[i]).to(torch.float32)

                if K=='input':
                    # reconstruct each same size as input
                    h_recon = torch.from_numpy(ot.unif(num_node_i))
                    K_recon = num_node_i
                else:
                    # reconstruct to a specified size K
                    h_recon = torch.from_numpy(ot.unif(K))
                    K_recon = K

                loss_i,T_i = self.mlp.fun_loss_cg(K_recon,h_recon,g_input,h_input,f_sample=f_sample)
                loss.append(loss_i)

            loss = torch.stack(loss)
            loss = torch.mean(loss)
            loss.backward()
            optim.step()
            optim.zero_grad()
            loss_l.append(loss.item())
            #print(str(epoch)+':'+str(loss.item()))

        return loss_l

    def get_W(self,K):
        W = self.mlp.sample(K)
        return W.detach().cpu().numpy()





        




