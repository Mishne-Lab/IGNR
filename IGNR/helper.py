import copy
import cv2
import numpy as np
import torch
import ot
from typing import List

'''
Note: The experiment set (of learning a single graphon) up is very similar to that of Xu et al 2021 
"Learning graphons via structured Gromov- Wasserstein barycenters, AAAI";
with the small difference that the graph size per graphon is {50,77,105,133,161,188,216,244,272,300}.

The graphs (in the data folder) were generated using the implementation from Xu et al 2021. 
One can check out the original repo of Xu et al 2021., for more details: https://github.com/HongtengXu/SGWB-Graphon

The part on using proximal gradient to estimate GW is also referenced from above
'''



'''
1. Computing GW using pg
'''

def proximal_ot(cost: np.ndarray,
                p1: np.ndarray,
                p2: np.ndarray,
                iters: int,
                beta: float,
                error_bound: float=1e-10,
                prior: np.ndarray=None) -> np.ndarray:
    """
    Borrow from Xu et al 2021:

    min_{T in Pi(p1, p2)} <cost, T> + beta * KL(T | prior)

    :param cost: (n1, n2) cost matrix
    :param p1: (n1, 1) source distribution
    :param p2: (n2, 1) target distribution
    :param iters: the number of Sinkhorn iterations
    :param beta: the weight of proximal term
    :param error_bound: the relative error bound
    :param prior: the prior of optimal transport matrix T, if it is None, the proximal term degrades to Entropy term
    :return:
        trans: a (n1, n2) optimal transport matrix
    """
    if prior is not None:
        kernel = np.exp(-cost / beta) * prior
    else:
        kernel = np.exp(-cost / beta)

    relative_error = np.inf
    a = np.ones(p1.shape) / p1.shape[0]
    b = []
    i = 0

    while relative_error > error_bound and i < iters:
        b = p2 / (np.matmul(kernel.T, a))
        a_new = p1 / np.matmul(kernel, b)
        relative_error = np.sum(np.abs(a_new - a)) / np.sum(np.abs(a))
        a = copy.deepcopy(a_new)
        i += 1
    trans = np.matmul(a, b.T) * kernel
    return trans

def node_cost_st(cost_s: np.ndarray, cost_t: np.ndarray, p_s: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """
    Borrow from Xu et al 2021:

    Calculate invariant cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
    Returns:
        cost_st: (n_s, n_t) array, the estimated invariant cost between the nodes in two graphs
    """
    n_s = cost_s.shape[0]
    n_t = cost_t.shape[0]
    f1_st = np.repeat((cost_s ** 2) @ p_s, n_t, axis=1)
    f2_st = np.repeat(((cost_t ** 2) @ p_t).T, n_s, axis=0)
    cost_st = f1_st + f2_st
    return cost_st


def gw_cost(cost_s: np.ndarray, cost_t: np.ndarray, trans: np.ndarray, p_s: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """
    Borrow from Xu et al 2021:

    Calculate the cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        trans: (n_s, n_t) array, the learned optimal transport between two graphs
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
    Returns:
        cost: (n_s, n_t) array, the estimated cost between the nodes in two graphs
    """
    cost_st = node_cost_st(cost_s, cost_t, p_s, p_t)
    return cost_st - 2 * (cost_s @ trans @ cost_t.T)
    

def gwloss_pg(W,g_data,h_recon,h_data,G0_prior=None,G0_cost=None,inner_iters=100,beta=5e-3):
    '''
    Compute T using proximal gradient, and return gradient with respect to W

    '''
    be = ot.backend.get_backend(W)
    p,q = ot.utils.list_to_array(h_recon,h_data)        
    p0, q0, C10, C20 = p, q, W, g_data
    p = be.to_numpy(p) # h_recon -- p
    q = be.to_numpy(q) # h_data -- q
    C1 = be.to_numpy(C10) # W -- C1
    C2 = be.to_numpy(C20) # g_data -- C2
    constC,hC1,hC2 = ot.gromov.init_matrix(C1,C2,p,q,loss_fun='square_loss')
    
    if G0_prior is None:
        G0_prior = p[:,None]*q[None,:]

    if G0_cost is None:
        G0_cost = p[:,None]*q[None,:]

    cost = gw_cost(C1,C2,G0_cost,np.expand_dims(p,1),np.expand_dims(q,1))
    cost /= np.max(cost)
    T = proximal_ot(cost,np.expand_dims(p,1),np.expand_dims(q,1),iters=inner_iters,beta=beta,prior=G0_prior)

    gwC1 = be.from_numpy(ot.gromov.gwloss(constC,hC1,hC2,T))
    gC1 = be.from_numpy(2 * C1 * (p[:, None] * p[None, :]) - 2 * T.dot(C2).dot(T.T))
    lossC1 = be.set_gradients(gwC1,C10,gC1)

    return lossC1,T


'''
2. Evaluation Helpers
'''
def gw_distance(graphon: np.ndarray, estimation: np.ndarray) -> float:
    p = np.ones((graphon.shape[0],)) / graphon.shape[0]
    q = np.ones((estimation.shape[0],)) / estimation.shape[0]
    loss_fun = 'square_loss'
    dw2 = ot.gromov.gromov_wasserstein2(graphon, estimation, p, q, loss_fun, log=False, armijo=False)
    return np.sqrt(dw2)


def mse_sort(graphon,W):
	ind = np.argsort(np.sum(W,axis=0))
	ind = ind[::-1]
	return np.linalg.norm(graphon - W[np.ix_(ind,ind)])



def get_graphs(graphs):
	# no aligning of sorting, just result the uniform weigth for the graphs

	num_nodes = [graphs[i].shape[0] for i in range(len(graphs))]
	max_num = max(num_nodes)
	min_num = min(num_nodes)

	all_graphs  = graphs
	all_weights = []
	for i in range(len(graphs)):
		num_i = graphs[i].shape[0]
		all_weights.append(np.expand_dims(ot.unif(num_i),axis=1))

	return all_graphs,all_weights,max_num,min_num

def gw_distanceG0(graphon: np.ndarray, estimation: np.ndarray) -> float:
    p = np.ones((graphon.shape[0],)) / graphon.shape[0]
    q = np.ones((estimation.shape[0],)) / estimation.shape[0]
    T0 = np.eye(graphon.shape[0],dtype=np.float32)/1000
    loss_fun = 'square_loss'
    dw2 = ot.gromov.gromov_wasserstein2(graphon, estimation, p, q, loss_fun, G0=T0, log=False, armijo=False)
    return np.sqrt(dw2)


'''
3. Graphon/Graph Similator; Borrow from Xu et al., 2021
'''
def synthesize_graphon(r: int = 1000, type_idx: int = 0) -> np.ndarray:
    """
    Synthesize graphons
    :param r: the resolution of discretized graphon
    :param type_idx: the type of graphon
    :return:
        w: (r, r) float array, whose element is in the range [0, 1]
    """
    u = ((np.arange(0, r) + 1) / r).reshape(-1, 1)  # (r, 1)
    v = ((np.arange(0, r) + 1) / r).reshape(1, -1)  # (1, r)

    if type_idx == 0:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = u @ v
    elif type_idx == 1:
        w = np.exp(-(u ** 0.7 + v ** 0.7))
    elif type_idx == 2:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 0.25 * (u ** 2 + v ** 2 + u ** 0.5 + v ** 0.5)
    elif type_idx == 3:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 0.5 * (u + v)
    elif type_idx == 4:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 1 / (1 + np.exp(-2 * (u ** 2 + v ** 2)))
    elif type_idx == 5:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 1 / (1 + np.exp(-(np.maximum(u, v) ** 2 + np.minimum(u, v) ** 4)))
    elif type_idx == 6:
        w = np.exp(-np.maximum(u, v) ** 0.75)
    elif type_idx == 7:
        w = np.exp(-0.5 * (np.minimum(u, v) + u ** 0.5 + v ** 0.5))
    elif type_idx == 8:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = np.log(1 + 0.5 * np.maximum(u, v))
    elif type_idx == 9:
        w = np.abs(u - v)
    elif type_idx == 10:
        w = 1 - np.abs(u - v)
    elif type_idx == 11:
        r2 = int(r / 2)
        w = np.kron(np.eye(2, dtype=int), 0.8 * np.ones((r2, r2)))
    elif type_idx == 12:
        r2 = int(r / 2)
        w = np.kron(np.eye(2, dtype=int), np.ones((r2, r2)))
        w = 0.8 * (1 - w)
    else:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = u @ v

    return w


def simulate_graphs(w: np.ndarray, seed_gsize: int=123, seed_edge:int=123, num_graphs: int = 10,
                    num_nodes: int = 200, graph_size: str = 'fixed') -> List[np.ndarray]:
    """
    Simulate graphs based on a graphon
    :param w: a (r, r) discretized graphon
    :param num_graphs: the number of simulated graphs
    :param num_nodes: the number of nodes per graph
    :param graph_size: fix each graph size as num_nodes or sample the size randomly as num_nodes * (0.5 + uniform)
    :return:
        graphs: a list of binary adjacency matrices
    """
    graphs = []
    r = w.shape[0]
	
    if graph_size == 'vary':
        numbers = np.linspace(50,300,num_graphs).astype(int).tolist()

    else: # fixed size
        numbers = [num_nodes for _ in range(num_graphs)]
    #print(numbers)
    
    np.random.seed(seed_edge) #add random seed for reproducibility
    for n in range(num_graphs):
        node_locs = (r * np.random.rand(numbers[n])).astype('int')
        graph = w[node_locs, :]
        graph = graph[:, node_locs]
        noise = np.random.rand(graph.shape[0], graph.shape[1])
        graph -= noise
        graphs.append((graph > 0).astype('float'))

    return graphs



















