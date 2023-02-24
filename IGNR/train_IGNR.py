import ot
from ot.gromov import gromov_wasserstein2
import numpy as np

from model_IGNR import *
from helper import *

import time
import pickle
import argparse

import copy
import os

'''
Goal: graphon learning model with SIREN:

Note: The experiment set up is very similar to that of Xu et al 2021 "Learning graphons via structured Gromov- Wasserstein barycenters";
with the small difference that the graph size per graphon is {50,77,105,133,161,188,216,244,272,300]}.

Most of the experimental set up code is referenced from Xu et al 2021. One can check out the original repo of Xu et al 2021. for more details:
https://github.com/HongtengXu/SGWB-Graphon


'''

parser = argparse.ArgumentParser()
parser.add_argument('--n-epoch', type=int, default=70,
                    help='number of traning epochs')
parser.add_argument('--use-pg', type=int, default=1,
                    help='1--use pg; 0--use cg')
parser.add_argument('--f-sample', type=str, default='fixed',
                    help='grid sampling strategy')
parser.add_argument('--w0', type=float, default=30,
                    help='default frequency for sine activation')
parser.add_argument('--mlp_dim_hidden', dest='mlp_dim_hidden', type=str, default="20,20,20",
					help='hidden units per layer for SIREN') 
parser.add_argument('--f-name', type=str, default='00',
					help='saving directory name')
args = parser.parse_args()


#Load all saved synthetic graphs of different sizes corresponding to the 0-12 diffrent graphons
ppath = os.getcwd()
with open(ppath+'/Data/graphs.pkl', 'rb') as f:
	data=pickle.load(f)


# matrix for collecting results
n_trials = len(data[0])
n_funs    = len(data)

error = np.zeros((n_funs,n_trials)) # gw error
error_mse = np.zeros((9,n_trials))  # mse error; those 9 graphons can be sorted by degrees

time_mat = np.zeros((n_funs,n_trials))
loss_mat = np.zeros((n_funs,args.n_epoch))


# define savepath
save_path = ppath+'/Result/'+args.f_name+'/'
if not os.path.exists(save_path):
  os.makedirs(save_path)
print('saving path is:'+save_path)


# Set experiemnt index to run 
exp_inds = [0,1]#range(13)

# Get f-theta architecture; convert to input format
args.mlp_dim_hidden = [int(x) for x in args.mlp_dim_hidden.split(',')]

for i_exp in exp_inds:

	graphon0 = synthesize_graphon(r=1000, type_idx=i_exp) #ground-truth graphon sampled at resolution 1000
	np.fill_diagonal(graphon0,0.) #ignore diagonal entries when computing GW error

#	err_best = np.inf

	for i_trial in range(n_trials):

		graphs = data[i_exp][i_trial]
		all_graphs = graphs

		if args.use_pg==1:
			gl_mlp = IGNR_pg_wrapper(args.mlp_dim_hidden,w0=args.w0)
			loss = gl_mlp.train(all_graphs,K='input',n_epoch=args.n_epoch,f_sample=args.f_sample) #n_epochs = 80?
		else:
			gl_mlp = IGNR_cg_wrapper(args.mlp_dim_hidden,w0=args.w0)
			loss = gl_mlp.train(all_graphs,K='input',n_epoch=args.n_epoch,f_sample=args.f_sample) #n_epochs = 80?


		#print(loss[-1])
		W1 = gl_mlp.get_W(1000) #get estimated graphon at resolution 1000
		if i_exp<=8:
			error_mse[i_exp,i_trial]=mse_sort(graphon0,W1)
		tmp_gw = gw_distance(graphon0,W1)

		error[i_exp,i_trial]=tmp_gw
		print('Data {}\tTrial {}\tError={:.3f}\t'.format(
                i_exp, i_trial, error[i_exp,i_trial]))



		# if tmp_gw < err_best:
		# 	err_best= tmp_gw
		# 	best_mlp = copy.deepcopy(gl_mlp_no.mlp.state_dict())
		# 	best_loss = copy.deepcopy(loss)


	#torch.save(best_mlp,save_path+'/exp'+str(i_exp)+'.pt')
	#torch.save(best_loss,save_path+'/loss'+str(i_exp)+'.pt')


np.set_printoptions(suppress=True)
print(np.round(np.mean(error[exp_inds,:], axis=1),3))
print(np.round(np.std(error[exp_inds,:], axis=1),3))



with open(save_path+'gw_pg'+str(args.use_pg)+'.pkl', 'wb') as f:
   pickle.dump(error, f)
with open(save_path+'mse_pg'+str(args.use_pg)+'.pkl', 'wb') as f:
   pickle.dump(error_mse, f)









