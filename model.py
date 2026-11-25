"""
Message-Passing GNNs from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - edges_to_coo
def edges_to_coo(edge_list, num_nodes=None):
    # TODO: Convert a list of (src, dst) edge pairs into COO-format src/dst tensors.
    if isinstance(edge_list,list):
        if len(edge_list)==0:
            edges=torch.empty((0,2),dtype=torch.long)

        else:
            edges=torch.tensor(edge_list,dtype=torch.long)
    else:
        edges=edge_list.long()

    src=edges[:,0]
    dst=edges[:,1]

    if num_nodes is None:
        if edges.shape[0]==0:
            num_nodes=0
        else:
            num_nodes=torch.max(edges).item()+1

    return src,dst,num_nodes

# Step 2 - add_self_loops
def add_self_loops(src, dst, num_nodes):
    """Append self-loop edges (i, i) for every node to COO edge indices.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes in the graph.

    Returns:
        src_out: LongTensor [E + num_nodes]
        dst_out: LongTensor [E + num_nodes]
    """
    # TODO: Append self-loop edges (i, i) for every node to the COO tensors
    self_idx=torch.arange(num_nodes,dtype=src.dtype)
    
    new_src=torch.cat([src,self_idx])
    new_dst=torch.cat([dst,self_idx])
    return new_src,new_dst

# Step 3 - compute_node_degrees
def compute_node_degrees(src, dst, num_nodes, edge_weight=None):
    """Compute per-node in-degrees (optionally weighted) from COO edges.

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].

    Returns:
        FloatTensor: In-degrees of shape [N].
    """
    # TODO: Compute per-node in-degrees by scattering onto destination nodes
    degrees=torch.bincount(dst,weights=edge_weight,minlength=num_nodes)
    return degrees.float()

# Step 4 - symmetric_normalize_edge_weights
def symmetric_normalize_edge_weights(src, dst, num_nodes, edge_weight=None):
    """Compute symmetrically normalized edge weights w_ij / sqrt(d_i * d_j).

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].
            Defaults to all ones (float32) when None.

    Returns:
        FloatTensor: Symmetrically normalized weights of shape [E].
    """
    # TODO: Compute symmetrically normalized edge weights for GCN-style propagation.
    E=src.shape[0]

    if edge_weight is None:
        edge_weight=torch.ones(E,dtype=torch.float32,device=src.device)
    
    degress=compute_node_degrees(src,dst,num_nodes,edge_weight)

    deg_inv_sqrt=torch.where(
        degress>0,
        torch.pow(degress,-0.5),
        torch.zeros_like(degress)
    )   
    src_factor=deg_inv_sqrt[src]
    dst_factor=deg_inv_sqrt[dst]

    normalized=edge_weight*src_factor*dst_factor
    return normalized

# Step 5 - gather_source_node_features
def gather_source_node_features(node_features, src):
    # TODO: Return edge-aligned source feature rows (E, F) from node_features.

    return  node_features[src]

# Step 6 - scatter_sum_to_nodes
def scatter_sum_to_nodes(edge_features, dst, num_nodes):
    """Scatter-sum edge features onto destination nodes to produce per-node aggregated vectors.

    Args:
        edge_features: FloatTensor of shape (E, F) with one feature row per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (N, F); row j is the sum of edge features with dst == j.
    """
    # TODO: Scatter-sum edge features onto destination nodes to produce per-node vectors
    
    F=edge_features.shape[1]
    out=torch.zeros((num_nodes,F),dtype=edge_features.dtype,device=edge_features.device)
    
    return out.index_add_(0,dst,edge_features)

# Step 7 - scatter_mean_to_nodes
def scatter_mean_to_nodes(edge_features, dst, num_nodes):
    # TODO: Scatter-mean edge features onto destination nodes (sum then divide by in-degree).
    
    
    sums=scatter_sum_to_nodes(edge_features,dst,num_nodes)

    counts=torch.bincount(dst,minlength=num_nodes).float()

    safe_counts=torch.clamp(counts,min=1)

    return sums/safe_counts.unsqueeze(1)

# Step 8 - scatter_max_to_nodes
def scatter_max_to_nodes(edge_features, dst, num_nodes):
    # TODO: Scatter-max edge features onto destination nodes (elementwise max).
    F=edge_features.shape[1]
    out=torch.full(
        (num_nodes,F),
        float('-inf'),
        dtype=edge_features.dtype,
        device=edge_features.device
    )
    dst_expanded=dst.unsqueeze(1).expand(-1,F)
    out.scatter_reduce_(
        dim=0,
        index=dst_expanded,
        src=edge_features,
        reduce='max'
    )

    return out

# Step 9 - compute_messages
def compute_messages(node_features, src, dst, message_fn, edge_attr=None):
    """Build per-edge messages via gather + message_fn.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        messages: FloatTensor of shape (E, M).
    """
    # TODO: Build per-edge messages by gathering features and applying message_fn
    

    src_feats=gather_source_node_features(node_features,src)
    dst_feats=node_features[dst]
    if edge_attr is not None:
        messages=message_fn(src_feats,dst_feats,edge_attr)
    else:
        messages=message_fn(src_feats,dst_feats)
    return messages

# Step 10 - aggregate_messages
def aggregate_messages(messages, dst, num_nodes, aggr='sum'):
    """Aggregate edge messages onto destination nodes using sum, mean, or max.

    Args:
        messages: FloatTensor of shape (E, M) with one message vector per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.
        aggr: str in {'sum', 'mean', 'max'} selecting the reduction.

    Returns:
        FloatTensor of shape (N, M); row j is the aggregated message for node j.
    """
    # TODO: Aggregate edge messages onto destination nodes via sum/mean/max...
    

    if aggr=='sum':
       
       return scatter_sum_to_nodes(messages,dst,num_nodes)

    elif aggr=='mean':
        
        return scatter_mean_to_nodes(messages,dst,num_nodes)
       
    elif aggr=='max':

        return  scatter_max_to_nodes(messages,dst,num_nodes)

# Step 11 - update_node_features
def update_node_features(node_features, aggregated, update_fn):
    # TODO: Implement update_node_features to fuse each node's current state with its aggregated...
    
    return update_fn(node_features,aggregated)

# Step 12 - message_passing_layer
def message_passing_layer(node_features, src, dst, message_fn, update_fn, aggr='sum', edge_attr=None):
    """Run one full Gilmer MPNN step: message, aggregate, and update.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages (E, M).
        update_fn: callable(node_features, aggregated) -> updated (N, H).
        aggr: str in {'sum', 'mean', 'max'}.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        updated_features: FloatTensor of shape (N, H).
    """
    # TODO: compose message, aggregate, and update into one MPNN step
    
    num_nodes=node_features.shape[0]

    messages=compute_messages(node_features,src,dst,message_fn,edge_attr=edge_attr)
    
    aggregated=aggregate_messages(messages,dst,num_nodes,aggr=aggr)
   
     
    result=update_node_features(node_features,aggregated,update_fn)
    
    return result

# Step 13 - stack_message_passing_layers
def stack_message_passing_layers(node_features, src, dst, layers, edge_attr=None):
    """Apply a sequence of message-passing layer callables to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        layers: list of callables, each
            layer(node_features, src, dst, edge_attr=None) -> Tensor (N, H_i).
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        embeddings: FloatTensor of shape (N, H), final layer output.
        all_layer_outputs: list of FloatTensors, one per layer (N, H_i).
    """
    # TODO: Apply a sequence of MP layer callables; return final + intermediates
    intermediate_outputs=[]
    current_features=node_features

    if not layers:
        return current_features,intermediate_outputs

    for layer in layers:
         current_features=layer(current_features,src,dst,edge_attr)
         intermediate_outputs.append(current_features)
    return current_features,intermediate_outputs

# Step 14 - gcn_renormalize_adjacency
def gcn_renormalize_adjacency(src, dst, num_nodes,edge_weight=None):
    """Apply Kipf-Welling renormalization: self-loops then symmetric norm.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes N.

    Returns:
        src_hat: LongTensor [E + N] sources after self-loops.
        dst_hat: LongTensor [E + N] destinations after self-loops.
        norm_weight: FloatTensor [E + N] symmetrically normalized weights.
    """
    # TODO: add self-loops then symmetrically normalize the adjacency...
    
    src_aug,dst_aug=add_self_loops(src,dst,num_nodes)
    weights_norm=symmetric_normalize_edge_weights(src_aug,dst_aug,num_nodes,edge_weight)

    return  src_aug,dst_aug,weights_norm

# Step 15 - gcn_linear_transform
def gcn_linear_transform(node_features, weight, bias=None):
    """Apply the GCN linear feature transform X @ W (+ bias).

    Args:
        node_features: FloatTensor of shape (N, Fin).
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout).

    Returns:
        FloatTensor of shape (N, Fout).
    """
    # TODO: compute the matrix product and optionally add a bias vector

    out=node_features@weight
    if bias is not None:
        out=out+bias

    return out

# Step 16 - gcn_layer_forward
def gcn_layer_forward(node_features, src, dst, weight, bias=None, num_nodes=None, activation=None):
    """Forward pass of one GCN layer: renormalize, transform, propagate.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; defaults to node_features.shape[0].
        activation: optional callable applied to the output.

    Returns:
        FloatTensor of shape (N, Fout).
    """
    # TODO: Forward pass of one GCN layer: renormalize, transform, propagate...
    if num_nodes is None:
        num_nodes=node_features.shape[0]
    # 归一化邻接矩阵
    src_new,dst_new,norm_weights=gcn_renormalize_adjacency(src,dst,num_nodes)
    # 线性变换
    transformed_features=gcn_linear_transform(node_features,weight,bias)
    # 提取节点特征
    src_feats=gather_source_node_features(transformed_features,src_new)
    # 乘以归一化权重
    weight_feats=src_feats*norm_weights.unsqueeze(1)
    
    agg_features=scatter_sum_to_nodes(weight_feats,dst_new,num_nodes)
    if activation is not None:
          agg_features=activation(agg_features)
    return agg_features

# Step 17 - init_gcn_parameters
import torch
import math

def init_gcn_parameters(in_dim, out_dim, with_bias=True, seed=None):
    # TODO: Initialize GCN weight (and optional bias) with Glorot-style uniform...
    if seed is not None:
        torch.manual_seed(seed)
    
    a=math.sqrt(6.0/(in_dim+out_dim))

    weight=torch.empty(in_dim,out_dim,dtype=torch.float32).uniform_(-a,a)

    result={'weight':weight}

    if with_bias:
        bias=torch.zeros(out_dim,dtype=torch.float32)
        result['bias']=bias
    return  result

# Step 18 - gcn_stack_forward (not yet solved)
# TODO: implement

# Step 19 - gat_attention_logits (not yet solved)
# TODO: implement

# Step 20 - gat_masked_neighbor_softmax (not yet solved)
# TODO: implement

# Step 21 - gat_head_forward (not yet solved)
# TODO: implement

# Step 22 - merge_gat_heads (not yet solved)
# TODO: implement

# Step 23 - gat_layer_forward (not yet solved)
# TODO: implement

# Step 24 - init_gat_parameters (not yet solved)
# TODO: implement

# Step 25 - gat_stack_forward (not yet solved)
# TODO: implement

# Step 26 - global_mean_pool (not yet solved)
# TODO: implement

# Step 27 - global_sum_pool (not yet solved)
# TODO: implement

# Step 28 - global_max_pool (not yet solved)
# TODO: implement

# Step 29 - global_mean_max_pool (not yet solved)
# TODO: implement

# Step 30 - node_classification_head (not yet solved)
# TODO: implement

# Step 31 - graph_regression_head (not yet solved)
# TODO: implement

# Step 32 - generate_sbm_graph (not yet solved)
# TODO: implement

# Step 33 - build_node_classification_dataset (not yet solved)
# TODO: implement

# Step 34 - generate_molecule_like_graph (not yet solved)
# TODO: implement

# Step 35 - build_graph_regression_dataset (not yet solved)
# TODO: implement

# Step 36 - collate_graph_batch (not yet solved)
# TODO: implement

# Step 37 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 38 - mse_loss (not yet solved)
# TODO: implement

# Step 39 - accuracy_metric (not yet solved)
# TODO: implement

# Step 40 - mae_metric (not yet solved)
# TODO: implement

# Step 41 - gnn_train_step (not yet solved)
# TODO: implement

# Step 42 - train_node_classifier (not yet solved)
# TODO: implement

# Step 43 - train_graph_regressor (not yet solved)
# TODO: implement

# Step 44 - representation_similarity (not yet solved)
# TODO: implement

# Step 45 - oversmoothing_diagnostic (not yet solved)
# TODO: implement

# Step 46 - mpnn_gnn_experiment (not yet solved)
# TODO: implement

