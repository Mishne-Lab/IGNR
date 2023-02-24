import networkx as nx
import numpy as np
import torch


from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_undirected
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch

import matplotlib.pyplot as plt

import gudhi  as gd

def pd_plot(rA,dim=2):
    skel = gd.RipsComplex(distance_matrix=1-rA,max_edge_length=1)
    rips_simplex = skel.create_simplex_tree(max_dimension=dim)
    rips_bar = rips_simplex.persistence()

    gd.plot_persistence_diagram(rips_bar)

def dendro_ind(adj,meth='single'):
    #adj: reconstructed probability adjacency matrix
    tmp = 1-adj
    np.fill_diagonal(tmp,0.)
    Y = sch.linkage(squareform(tmp),meth)
    ind = sch.leaves_list(Y)
    return ind

def dendro_ind_D(D,meth='single'):
    #adj: reconstructed probability adjacency matrix
    tmp = D
    np.fill_diagonal(tmp,0.)
    Y = sch.linkage(squareform(tmp),meth)
    ind = sch.leaves_list(Y)
    return ind

def n_community(c_sizes, p_inter=0.01, p_intra=0.7,flag_perm=False):
    '''
    Function to generate n_community graph
    '''
    order = []
    graphs = [nx.gnp_random_graph(c_sizes[i], p_intra) for i in range(len(c_sizes))]
    #graphs = [nx.gnp_random_graph(c_sizes[i], p_intra, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = [G.subgraph(c) for c in nx.connected_components(G)]
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i+1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])

    if flag_perm:
        adj = nx.to_numpy_matrix(G)
        ind = np.random.permutation(adj.shape[0])
        adj = adj[np.ix_(ind,ind)]
        G=nx.from_numpy_matrix(adj)
        order = ind.argsort()

    return G




def rGraphData_tg2(G_data,name_data,features='row_id'):

    print('using '+features+' as node feature')
    
    M = len(G_data[1])
    data_list = []
    max_num_nodes = np.max(G_data[1])
    for i in np.arange(M):
        N = G_data[1][i]

        edge_index = to_undirected(torch.tensor(np.array(G_data[0][i]).T))
        
        if features == 'row_id':
            node_features = torch.tensor(np.arange(N),dtype=torch.float32).unsqueeze(1)
        elif features in ['attr_discrete']:
            node_features = torch.tensor(G_data[4][i],dtype=torch.float32).unsqueeze(1)
        elif features in ['attr_real']:
            node_features = torch.tensor(G_data[4][i],dtype=torch.float32)
        else:
            raise ValueError('feature not specified')

        order = np.arange(N)      
        data = Data(x=node_features,edge_index=edge_index,order=order)
        data_list.append(data)
            
        input_dim = node_features.size()[1]

    if features == 'row_id':
        n_card = max_num_nodes
    elif features in ['attr_discrete']:
        n_card = G_data[5]
    elif features in ['attr_real']:
        n_card = input_dim
    else:
        raise ValueError('This dataset not specified')

    return data_list,input_dim,n_card




def GraphData_tg(G_list,features='row_id',add_perm=True):
    '''
    Function that convert networkx graph list to pytorch-geometric Data format

    '''
    
    # The Data list to return
    data_list = []
    max_num_nodes = 0
    
    coor_flag=False
    if len(G_list[0])==2:
        coor_flag=True
        
    for i in np.arange(len(G_list)):
        
        if coor_flag:
            adj = nx.to_numpy_array(G_list[i][0])
            P = G_list[i][1]
        else:    
            # get adj from graph list
            adj = nx.to_numpy_array(G_list[i])
        
        N = adj.shape[0]
        if N>max_num_nodes:
            max_num_nodes=N

        # add permutation to input adj matrix
        order = np.arange(adj.shape[0])
        if add_perm:
            ind = np.random.permutation(adj.shape[0])
            order = ind.argsort()
            adj = adj[np.ix_(ind,ind)]

        # get edge list
        edge_index = dense_to_sparse(torch.tensor(adj))[0]      
        

        # expand node feature
        if features == 'id':
            node_features = torch.tensor(np.identity(N),dtype=torch.float32)
            n_card = max_num_nodes
            
        elif features == 'row_id':
            node_features = torch.tensor(np.arange(N),dtype=torch.float32).unsqueeze(1)
            n_card = max_num_nodes
            
        elif features == 'deg':
            # heuristic of using one-hot encoding of deg as node feature
            node_features = torch.zeros(N, input_dim)
            # padded ones has vector 0 for node feature
            node_features[range(N), [tag2index[tag] for tag in np.sum(A,1)]] = 1
            n_card = np.unique(np.sum(A,1))
            
        elif features == 'const':
            #same node feature size as "deg", but with all entries the same, 1
            node_features = torch.ones(N, 1)
            n_card = 2
            
        if coor_flag:
            data = Data(x=node_features,edge_index=edge_index,order=order,coor=P)
        else:
            data = Data(x=node_features,edge_index=edge_index,order=order)
        
        data_list.append(data)
    
    input_dim = node_features.size()[1]
        
    return data_list, input_dim, n_card



class Graphon(object):
    def __init__(self, graphon):
        '''
        graphon: function that takes 2 arguments in [0,1], returns value in [0,1]
        '''
        self.graphon = graphon

    def sample(self, N, seed_adj=None):
        '''
        Samples a graph of N vertices from the graphon.
        N: graph size
        seed_wts: seed for uniform edge
        seed_adj: seed for sampling from Bernouli
        '''
        

        U = (np.arange(N)+(1/2))/N #deterministic index
        #U = np.random.uniform(0,1,N) #sampled index
        wts = np.array([[self.graphon(U[i], U[j]) for i in range(N)] for j in range(N)])
        
        np.random.seed(seed_adj)
        A = (np.random.uniform(0,1,(N,N))<np.triu(wts,1))*1 #assuming wts to be symmetric
        
        A = A+A.T
        order=U.argsort()
  
        #return A,wts,order
        return nx.from_numpy_matrix(A)
        
    def plot(self, N=200, colorbar=False, wts=None, save=None, title=None):
        #from pygraphon.core.graphon_utils import plot_graphon
        grid = np.arange(0,1,1/N)
        wts  = np.array([[self.graphon(grid[i], grid[j]) for i in range(N)] for j in range(N)])
        plt.imshow(0.5*(wts+wts.T))
        plt.colorbar()
        plt.clim([0,1])
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.axis('off')
    
    def get_wts(self,N=1000):
        grid = (np.arange(N)+(1/2))/N
        wts  = np.array([[self.graphon(grid[i], grid[j]) for i in range(N)] for j in range(N)])
        wts  = np.triu(wts,1)
        wts  = wts+wts.T
        np.fill_diagonal(wts,0.)
        return wts

def SBM3(p1=0.9,r1=2/3,r2=1/2):
    #simple SBM with only three blobs; center lines define by c1 and c2, within p is p1, outside p is p2
    # p1: intra class probability
    # r1: ratio 1
    # r2: ratio 2
    
    p2 = 1-p1
    c2 = r1
    c1 = r1*r2
    return lambda x,y: p1 if (x<=c1 and y<c1) or (c1<x<=c2 and c1<=y<c2) or (c2<x and c2<y) else p2 


def SBM2(p1=0.9,p2=0.1,r1=0.5):
    # p1: inner class proba
    # r1: ratio for the two blocks
    c1 = r1
    return lambda x,y: p1 if (x<=c1 and y<=c1) or (x>c1 and y>c1) else p2

def ER(p1=0.9):
    return lambda x,y: p1

def gfun(t):
    # A "cycle" like graphon with changing width
    theta = np.pi*3/4
    return lambda x,y: 0.9*np.exp((-(y)**2-(x-1)**2)/t**2)+0.9*np.exp(-((np.sin(theta)*x+np.cos(theta)*y)/t)**2)#np.clip(1-np.abs(x-y-0.4)*t,0,1)

