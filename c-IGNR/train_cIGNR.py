import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import copy

import torch

from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import ot
from ot.gromov import gromov_wasserstein2
import cv2

# --- Model ---
from model_cIGNR import cIGNR

from siren_pytorch import *

from evaluation import *

from data import rGraphData_tg2,GraphData_tg

from sklearn.metrics import rand_score
from sklearn.cluster import KMeans
import time



'''
Traning Function
'''

def train(args, train_loader, model, test_loader):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.step_size),
        gamma=float(args.gamma)
    )

    loss_list = []
    loss_list_batch = []


    tmp_list = []


    best_loss_batch  = np.inf
    best_model = None
    #best_C_recon_test = []

    best_acc = 0.
    acc_list = []
    
    since = time.time()

    for epoch in range(args.n_epoch):

        model.train()
        
        loss_epoch = []

        for batch_idx, data in enumerate(train_loader):

            x = data.x.to(torch.int64).to(args.device)            
            edge_index = data.edge_index.to(torch.int64).to(args.device)
            batch = data.batch.to(torch.int64).to(args.device)
            C_input = to_dense_adj(edge_index,batch=batch)

            # siren mlp
            loss,z,C_recon_list=model(x,edge_index,batch,C_input,args.M)

            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loss_epoch.append(loss.item())
            
            print(f'Epoch: {epoch:03d}, Batch: {batch_idx:03d}, Loss:{loss.item():.4f}')

        loss_batch = np.mean(loss_epoch) # this is not loss on test set, this is average training loss across batches
        loss_list_batch.append(loss_batch)
        z = get_emb(args,model,test_loader)


        if loss_batch<best_loss_batch:
            best_loss_batch=loss_batch
            best_model=copy.deepcopy(model)
            best_z = get_emb(args,model,test_loader)
   
    finish = time.time()
    print('time used:'+str(finish-since))


    print('loss on per epoch:')
    print(loss_list_batch)


    # print('acc per epoch:')
    # print(acc_list)

    # save trained model and loss here
    print('Finshed Training')
    
    torch.save({'epoch': epoch, 
        'batch': batch_idx, 
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},args.save_path + 'end_loss_weights.pt')


    np.savetxt(args.save_path + 'loss.out',loss_list,fmt='%.4f',delimiter=',')
    np.savetxt(args.save_path + 'loss_batch.out',loss_list_batch,fmt='%.4f',delimiter=',')
    #np.savetxt(args.save_path + 'acc_list.out',acc_list,fmt='%.4f',delimiter=',')
    np.save(args.save_path+'z_best.npy',best_z)



    # plot loss here
    fig1 = plt.figure(figsize=(8,5))
    plt.plot(loss_list)
    fig1.savefig(args.save_path+'loss.png')
    plt.close(fig1)



def arg_parse():
    parser = argparse.ArgumentParser(description='GraphonAE arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')

    ### Optimization parameter
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--step_size', dest='step_size', type=float,
            help='Learning rate scheduler step size')
    parser.add_argument('--gamma', dest='gamma', type=float,
            help='Learning rate scheduler gamma')

    ### Training specific
    parser.add_argument('--n_epoch', dest='n_epoch', type=int,
            help='Number of training epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--cuda', dest='cuda', type=int,
            help='cuda device number')

    parser.add_argument('--feature', dest='feature',
            help='Feature used for encoder.')
    parser.add_argument('--save_dir', dest='save_dir',
            help='name of the saving directory')
    parser.add_argument('--flag_eval',dest='flag_eval',type=int,help='whether to compute graphon recon error') 

    # General model param
    parser.add_argument('--flag_emb',dest='flag_emb',type=int) 
    parser.add_argument('--gnn_num_layer', dest='gnn_num_layer', type=int)

    ### Model selection and sampling reconstruction number
    #parser.add_argument('--add_perm',dest='add_perm',type=bool)
    #parser.add_argument('--model_ind',dest='model_ind',type=int) #model number to select
    parser.add_argument('--M',dest='M',type=int) #sampling number for graph reconstruction if needed

    ### SIREN-MLP-model specific
    parser.add_argument('--mlp_dim_hidden', dest='mlp_dim_hidden') #hidden dim (number of neurons) for f_theta
    parser.add_argument('--emb_dim', dest='emb_dim', type=int)
    parser.add_argument('--latent_dim', dest='latent_dim', type=int) #from graph embedding to latent embedding, reducing graph embedding dimension
    parser.add_argument('--mlp_act', dest='mlp_act') # whether to use sine activation for the mlps


    ###

    parser.set_defaults(dataset='2ratio_rand',
                        feature='row_id',
                        lr=0.01,
                        n_epoch=12,
                        batch_size=10,
                        cuda=0,
                        save_dir='00',
                        step_size=4,
                        gamma=0.1,
                        gnn_num_layer=3,
                        latent_dim=16,
                        emb_dim=16,
                        mlp_dim_hidden='48,36,24',
                        mlp_act = 'sine',
                        flag_emb=1,
                        flag_eval=0,
                        M=0)
    return parser.parse_args()



def main():

    prog_args = arg_parse()

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(prog_args.cuda)
   
    torch.cuda.set_device(int(prog_args.cuda))
    prog_args.device = 'cuda:' + str(prog_args.cuda) if int(prog_args.cuda) >= 0 else 'cpu'
    print('CUDA', prog_args.cuda)

    ppath = os.getcwd()

    prog_args.save_path=ppath+'/Result/'+prog_args.save_dir+'/'
    if not os.path.exists(prog_args.save_path):
        os.makedirs(prog_args.save_path)
    print('saving path is:'+prog_args.save_path)


    # Load Specific Dataset
    with open(ppath+'/Data/'+prog_args.dataset+'.pkl','rb') as f:  
        data = pickle.load(f)


    if prog_args.dataset in ['IMDB-B','IMDB-M']:   
        G_data = data
        n_sample = len(G_data[0])
        print('total samples:'+str(n_sample))
        G_train = G_data #G_data[:n_train] training on all data for unsupervised embedding
        #G_test  = G_data
        gtlabel = G_train[2]
        train_dataset,_,n_card = rGraphData_tg2(G_train,prog_args.dataset,prog_args.feature) 
        test_dataset = train_dataset # obtain unsupervised embedding on whole dataset

    else: # For learning parameterized graphon on synthetic data

        G_data = data[0]
        labels = data[1]

        n_sample = len(G_data)
        n_train  = round(n_sample*.9)
        if np.mod(n_train,2) == 1:
            n_train = n_train+1
        print('total samples:'+str(n_sample)+'  train samples:'+str(n_train))
        G_train = G_data[:n_train]
        G_test  = G_data[n_train:]
        tlabel  = labels[n_train:] 

        train_dataset,_,n_card = GraphData_tg(G_train,features=prog_args.feature,add_perm=True)
        test_dataset,_,n_card = GraphData_tg(G_test,features=prog_args.feature,add_perm=True)


    train_loader = DataLoader(train_dataset, batch_size=prog_args.batch_size,shuffle=False) # Data is pre-shuffled by fixed seed
    test_loader  = DataLoader(test_dataset, batch_size=len(test_dataset),shuffle=False) # For evaluating and saving all embeddings

    prog_args.step_size = len(train_loader)*prog_args.step_size     
    prog_args.mlp_dim_hidden = [int(x) for x in prog_args.mlp_dim_hidden.split(',')]
    prog_args.mlp_num_layer = len(prog_args.mlp_dim_hidden)

    snet = SirenNet(
        dim_in = 2, # input [x,y] coordinate
        dim_hidden = prog_args.mlp_dim_hidden,
        dim_out = 1, # output graphon (edge) probability 
        num_layers = prog_args.mlp_num_layer, # f_theta number of layers
        final_activation = 'sigmoid',
        w0_initial = 30.,
        activation = prog_args.mlp_act)
    model = cIGNR(net=snet,input_card=n_card,emb_dim=prog_args.emb_dim,latent_dim=prog_args.latent_dim,num_layer=prog_args.gnn_num_layer,device=prog_args.device,flag_emb=prog_args.flag_emb)




    model = model.to(torch.device(prog_args.device))
    train(prog_args,train_loader,model, test_loader)

    #plot_eval(prog_args,model,train_dataset,9,'sample_train')


    if prog_args.flag_eval==1 and prog_args.dataset in ['gCircle','2ratio_rand']:
        print('Evaluating Graphon Matching Loss')
        # Note: evaluating Graphon Reconstruction can be slow due to comparing large resolution matrices
        compute_graphon_loss(prog_args,model,test_loader,tlabel)

if __name__ == '__main__':
    main()













