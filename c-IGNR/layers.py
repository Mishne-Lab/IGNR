import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops


# Note:
# - Baseline MLP decoder
# - GNN encoder layers



# Linear layer =============================================================
class decoder_MLP(nn.Module):
    '''
    MLP: linear layers to map from graph embedding to desired output 
    (e.g. edge probability, dictionary, dictionary function parameters etc)

    '''
    def __init__(self,dim_out,dim_in=32,dim_lat=[32,64],add_sig=True):
        super().__init__()
        self.dim_in  = dim_in
        self.dim_out = dim_out
        self.dim_lat = dim_lat

        n_layer = len(dim_lat)

        layers = []
        layers.append(nn.Linear(dim_in,dim_lat[0]))
        #layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(dim_lat[0]))
        layers.append(nn.ReLU())        
        for i_layer in range(n_layer-1):
            layers.append(nn.Linear(dim_lat[i_layer],dim_lat[i_layer+1]))
            #layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim_lat[i_layer+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim_lat[-1],dim_out))
        if add_sig:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    def forward(self,x):
        return self.net(x)


# GNN layers =================================================================
class SAGE_Conv(MessagePassing):
    """
    GraphSage
    """
    def __init__(self, input_dim, emb_dim, aggr = "max", input_layer = False):
        super(SAGE_Conv, self).__init__()
        # linear layer
        self.linear = torch.nn.Linear(emb_dim,emb_dim)
        self.act = torch.nn.ReLU()
        self.update_linear = torch.nn.Linear(2*emb_dim,emb_dim)
        self.update_act = torch.nn.ReLU()
        
        ### Mapping uniform input features to embedding.
        self.input_layer = input_layer
        if self.input_layer:
            self.input_node_embeddings = torch.nn.Embedding(input_dim, emb_dim)
            torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)
            
        self.aggr = aggr

    def forward(self, x, edge_index):
        
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))[0]          
        
        if self.input_layer:
            x = self.input_node_embeddings(x.to(torch.int64).view(-1,))
        
        return self.propagate(edge_index, aggr=self.aggr, x=x)

    def message(self, x_j):
        x_j = self.linear(x_j)
        x_j = self.act(x_j)
        return x_j

    def update(self, aggr_out, x):
        
        new_emb= torch.cat([aggr_out,x],dim=1)
        new_emb= self.update_linear(new_emb)
        new_emb= self.update_act(new_emb)
        #print('Here')
        #print(new_emb.size())
        return F.normalize(new_emb,p=2,dim=-1)
    
class GIN_Conv(MessagePassing):
    """
    GIN without edge feature concatenation
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        input_layer (bool): whethe the GIN conv is applied to input layer or not. (Input node labels are uniform...)
    """
    def __init__(self, input_dim, emb_dim, aggr = "add"):
        super(GIN_Conv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), 
                                       torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.aggr = aggr

    def forward(self, x, edge_index):
        
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))[0]
        
        return self.propagate(edge_index, aggr=self.aggr, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return self.mlp(aggr_out)
