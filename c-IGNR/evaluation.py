from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_undirected
import matplotlib.pyplot as plt
import numpy as np
import torch

import ot
from ot.gromov import gromov_wasserstein2
from data import *



def get_emb(args,model,test_loader):
    model.eval()
    for batch_idx, data in enumerate(test_loader):
        if args.feature in ['attr_real']:
            x = data.x.to(torch.float32).to(args.device)
        else:
            x = data.x.to(torch.int64).to(args.device)
        edge_index = data.edge_index.to(torch.int64).to(args.device)
        batch = data.batch.to(torch.int64).to(args.device)

    z = model.encode(x, edge_index, batch)
    return z.detach().cpu().numpy()



def plot_eval(args,model,test_dataset,n,sname):
    # evaluation plot for synthetic data, plotting the input sample and its weight on the dictionaries, in one row
    # n: number of samples to plot


    model.eval()
    test_l  = DataLoader(test_dataset[:n], batch_size=n, shuffle=False)

    for data in test_l:
        x = data.x.to(torch.int64).to(args.device)
        edge_index = data.edge_index.to(torch.int64).to(args.device)
        batch = data.batch.to(torch.int64).to(args.device)
        C_input = to_dense_adj(edge_index,batch=batch)
        # if permutation added, permute it back for better visualization
        #if args.add_perm:
        order = data.order

        g = model.encode(x,edge_index,batch)
        loss,z, C_recon_list = model(x,edge_index,batch,C_input,args.M)

    z_emb=z.detach().cpu().numpy().round(2)
    C_input = C_input.detach().cpu().numpy()

    fig=plt.figure(figsize=(2*1.5,1.5*n))
    for i in np.arange(n):
        plt.subplot(n,2,i*2+1)
        C_input_tmp=C_input[i,:,:]
        #if args.add_perm:
        tmp_ind = order[i]
        C_input_tmp=C_input_tmp[np.ix_(tmp_ind,tmp_ind)]
        plt.imshow(C_input_tmp)
        plt.axis('off')
        
        #Cr = model.decode_sample(z[i],args.M).detach().cpu().numpy()
        Cr = C_recon_list[i]
        plt.subplot(n,2,i*2+2)
        ind = dendro_ind(Cr)
        #plt.imshow(Cr[np.ix_(ind,ind)])
        plt.imshow(Cr)
        #plt.title(str(z_emb[i]))
        plt.colorbar()
        #plt.clim([0,0.2])
        plt.axis('off')
    fig.savefig(args.save_path+sname+'.png')
    plt.close(fig)



'''
Graphon Reconstruction evaluation for the two synthetic parameterized graphons:
'''
def compute_graphon_loss(args, model, test_loader, tlabel):
    '''
    tlabel: ground-truth parameter for the changing graphon
    '''

    model.eval()
    
    # Get reconstruction on held out testing set
    for batch_idx, data in enumerate(test_loader):
        x = data.x.to(torch.int64).to(args.device)
        edge_index = data.edge_index.to(torch.int64).to(args.device)
        batch = data.batch.to(torch.int64).to(args.device)
        C_input = to_dense_adj(edge_index,batch=batch)

        C_recon_list = model.sample(x,edge_index,batch,1000)


    N = C_input.shape[0]
    loss = []
    for i in np.arange(N):

        if args.dataset=='2ratio_rand':
            sbm = Graphon(SBM2(p1=0.9,p2=0.1,r1=tlabel[i]))
        elif args.dataset =='gCircle':
            sbm = Graphon(gfun(t=tlabel[i]))
        graphon_gt =sbm.get_wts(1000)

        loss.append(gromov_wasserstein2(C_recon_list[i],graphon_gt,ot.unif(C_recon_list[i].shape[0]),ot.unif(1000)))
        print(loss[i])


    loss_val = np.mean(loss)
    print('average:')
    print(loss_val)
    np.save(args.save_path + 'loss_graphon.npy',loss_val)
































