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

# Step 18 - gcn_stack_forward
def gcn_stack_forward(node_features, src, dst, param_list, activations=None, num_nodes=None):
    """Run a stack of GCN layers to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F0).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        param_list: list of dicts, each with 'weight' (Fin, Fout) and optional 'bias' (Fout,).
        activations: optional list of callables or None, one per layer.
        num_nodes: optional int N; defaults to node_features.shape[0].

    Returns:
        embeddings: FloatTensor of shape (N, FL), the final layer output.
        all_layer_outputs: list of FloatTensor outputs after each layer.
    """
    # TODO: Run a stack of GCN layers to produce deep node embeddings
    if num_nodes is None:
        num_nodes=node_features.shape[0]
    current_features=node_features
    all_layer_outputs=[]
    for i,param in enumerate(param_list):
        weight=param['weight']
        bias=param.get('bias',None)
        activation=activations[i] if activations is not None else None
        current_features=gcn_layer_forward(current_features,src,dst,weight,bias,num_nodes,activation)
        all_layer_outputs.append(current_features)
    return current_features,all_layer_outputs

# Step 19 - gat_attention_logits
import torch.nn.functional as F
def gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight):
    """Compute unnormalized GAT attention logits and transformed features.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.

    Returns:
        logits: FloatTensor of shape (E,) unnormalized attention scores.
        transformed: FloatTensor of shape (N, Fout) linearly transformed nodes.
    """
    # TODO: return per-edge LeakyReLU attention logits and transformed features
    
    H=gcn_linear_transform(node_features,weight,bias=None)
    H_src=H[src]
    H_dst=H[dst]
    
    result=F.leaky_relu(attn_src@H_src.T+attn_dst@H_dst.T,negative_slope=0.2)
    return result,H

# Step 20 - gat_masked_neighbor_softmax
def gat_masked_neighbor_softmax(logits, dst, num_nodes):
    """Numerically stable softmax of attention logits over each dest node's neighbors.

    Args:
        logits: FloatTensor of shape (E,) with one unnormalized attention logit per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (E,) with attention coefficients that sum to 1 over
        each destination's incoming edges.
    """
    # TODO: Numerically stable softmax of attention logits over each dest node's neighbors

    logits_2d=logits.unsqueeze(1)
    max_per_node=scatter_max_to_nodes(logits_2d,dst,num_nodes)
    max_per_edge=max_per_node[dst]
    exp_logits=torch.exp(logits_2d-max_per_edge)
    sum_per_node=scatter_sum_to_nodes(exp_logits,dst,num_nodes)
    sum_per_edge=sum_per_node[dst]
    alpha=(exp_logits/sum_per_edge).squeeze(1)

    return alpha

# Step 21 - gat_head_forward
def gat_head_forward(node_features, src, dst, weight, attn_src, attn_dst, bias=None, num_nodes=None, activation=None):
    """Forward pass of a single GAT attention head.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied to the head output.

    Returns:
        head_out: FloatTensor of shape (N, Fout).
        attn_coeffs: FloatTensor of shape (E,) attention coefficients.
    """
    # TODO: Forward pass of a single GAT attention head: transform, coeffs, aggregate...
    if num_nodes is None:
        num_nodes=node_features.shape[0]
    
    logits,H=gat_attention_logits(node_features,src,dst,attn_src,attn_dst,weight)
    
    alpha=gat_masked_neighbor_softmax(logits,dst,num_nodes)

    H_src=gather_source_node_features(H,src)
    
    weight_feats=H_src*alpha.unsqueeze(1)

    output=scatter_sum_to_nodes(weight_feats,dst,num_nodes)
 

    if bias is not None:
        output=output+bias
    if activation is not None:
        output=activation(output)
    return  output,alpha

# Step 22 - merge_gat_heads
def merge_gat_heads(head_outputs, mode='concat'):
    # TODO: Merge multi-head GAT outputs into one node-feature tensor.
    if isinstance(head_outputs,(list,tuple)):
        stacked=torch.stack(head_outputs,dim=0)
    else:
        stacked=head_outputs

    
    if mode=='concat':
      H,N,F=stacked.shape
      permuted=stacked.permute(1,0,2)
      return permuted.reshape(N,H*F)
    
    elif mode=='mean':
       return torch.mean(stacked,dim=0)
    else:
        raise ValueError(f'Unsupported mode:{mode}')

# Step 23 - gat_layer_forward
def gat_layer_forward(node_features, src, dst, head_params, merge_mode='concat', num_nodes=None, activation=None):
    """Multi-head GAT layer: run each head, merge, optional activation.

    Args:
        node_features: FloatTensor (N, Fin).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        head_params: list of dicts with keys weight, attn_src, attn_dst,
            and optional bias for each head.
        merge_mode: 'concat' or 'mean'.
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied after merging heads.

    Returns:
        out: FloatTensor (N, F_merged).
        all_attn: list of FloatTensor (E,) attention coeffs per head.
    """
    # TODO: run each head, merge outputs, apply optional nonlinearity...
    head_outputs=[]
    all_attn=[]
    if num_nodes is None:
           num_nodes=node_features.shape[0]
    for params in head_params:
        weight=params['weight']
        attn_dst=params['attn_dst']
        attn_src=params['attn_src']
        bias=params.get('bias',None)
        output_head,attn_head=gat_head_forward(node_features,src,dst,weight,attn_src,attn_dst,bias,num_nodes)
        head_outputs.append(output_head)
        all_attn.append(attn_head)
    merged_output=merge_gat_heads(head_outputs,mode=merge_mode)
    if activation is not None:
        merged_output=activation(merged_output)
    return merged_output,all_attn

# Step 24 - init_gat_parameters
import torch
import math
def init_gat_parameters(in_dim, out_dim, num_heads=1, with_bias=True, seed=None):
    # TODO: Initialize multi-head GAT parameters with Glorot-style initialization.
    if seed is not None:
        torch.manual_seed(seed)
    params_list=[]
    for _  in range(num_heads):
        a_w=math.sqrt(6/(in_dim+out_dim))
        a_a=math.sqrt(6/(out_dim+1))
        weight=torch.empty(in_dim,out_dim).uniform_(-a_w,a_w)
        weight.requires_grad=True
        attn_src=torch.empty(out_dim).uniform_(-a_a,a_a)
        attn_src.requires_grad=True
        attn_dst=torch.empty(out_dim).uniform_(-a_a,a_a)
        attn_dst.requires_grad=True
        head_dict={
            'weight':weight,
            'attn_src':attn_src,
            'attn_dst':attn_dst,
        }

        if with_bias:
            bias=torch.zeros(out_dim)
            bias.requires_grad=True
            head_dict['bias']=bias
        params_list.append(head_dict)

    return params_list

# Step 25 - gat_stack_forward
def gat_stack_forward(node_features, src, dst, layer_param_list, merge_modes=None, activations=None, num_nodes=None):
    """Run a stack of multi-head GAT layers.

    Args:
        node_features: FloatTensor (N, F0).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        layer_param_list: list of length L; each entry is a head_params list
            for gat_layer_forward.
        merge_modes: optional list of L merge mode strings ('concat' or 'mean').
            Defaults to 'concat' for every layer.
        activations: optional list of L callables or None. Defaults to no
            activation for every layer.
        num_nodes: optional int N; inferred from node_features if None.

    Returns:
        embeddings: FloatTensor (N, FL) final layer output.
        all_layer_outputs: list of L FloatTensors, the output after each layer.
    """
    # TODO: Run a stack of multi-head GAT layers for deep node embeddings.
    if num_nodes is None:
        num_nodes=node_features.shape[0]
    all_layer_outputs=[]
    current_features=node_features
    for i,head_params in enumerate(layer_param_list):
        mode=merge_modes[i] if  merge_modes is not None else 'concat'
        act=activations[i]  if activations is not None else None
        out,all_attn=gat_layer_forward(current_features,src,dst,head_params,merge_mode=mode,num_nodes=num_nodes,activation=act)
        current_features=out
        all_layer_outputs.append(current_features)

    return current_features,all_layer_outputs

# Step 26 - global_mean_pool
def global_mean_pool(node_features, batch_index, num_graphs=None):
    """Globally mean-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one feature row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id in
            {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, F); row b is the mean of node features with
        batch_index == b.
    """
    # TODO: Mean-pool node features into one graph-level vector per graph...
    if num_graphs is None:
       B=int(batch_index.max())+1
    else:
       B=num_graphs
    
    sum_features=scatter_sum_to_nodes(node_features,batch_index,B)
    counts=torch.bincount(batch_index,minlength=B).float()
    safe_counts=torch.clamp(counts,min=1.0)
    mean_features=sum_features/safe_counts.unsqueeze(1)
    return mean_features

# Step 27 - global_sum_pool
def global_sum_pool(node_features, batch_index, num_graphs=None):
    """Globally sum-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id
            in 0 .. B-1.
        num_graphs: optional int B. If None, inferred as max(batch_index) + 1.

    Returns:
        FloatTensor of shape (B, F); row g is the sum of node features with
        batch_index == g.
    """
    # TODO: sum-pool node features into one graph-level vector per graph
    if num_graphs is None:
       B=int(batch_index.max())+1

    else:
        B=num_graphs
   
    sum_features=scatter_sum_to_nodes(node_features,batch_index,B)
   
    return sum_features

# Step 28 - global_max_pool
def global_max_pool(node_features, batch_index, num_graphs=None):
    # TODO: Globally max-pool node features into one graph-level vector per graph.
    if num_graphs is None:
       B=int(batch_index.max())+1

    else:
        B=num_graphs

    max_nodes=scatter_max_to_nodes(node_features,batch_index,B)
    
    return max_nodes

# Step 29 - global_mean_max_pool
def global_mean_max_pool(node_features, batch_index, num_graphs=None):
    """Concatenate global mean and max pooled features into a 2F-dim graph vector.

    Args:
        node_features: FloatTensor of shape (N, F).
        batch_index: LongTensor of shape (N,) with graph ids in {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, 2F); each row is [mean_pool || max_pool].
    """
    # TODO: Concatenate global mean and max pooled features into a 2F-dim vector...
    if  num_graphs is None:
        B=int(batch_index.max())+1
    else:
        B=num_graphs
    
    mean_features=global_mean_pool(node_features,batch_index,B)
    max_features=global_max_pool(node_features,batch_index,B)

    result=torch.cat([mean_features,max_features],dim=-1)
    return result

# Step 30 - node_classification_head
def node_classification_head(node_embeddings, weight, bias=None):
    # TODO: Map node embeddings to per-node class logits via a linear head...
    result=node_embeddings@weight

    if bias is None:
        output=result
    else:
        output=result+bias
    return output

# Step 31 - graph_regression_head
def graph_regression_head(graph_embeddings, weight, bias=None):
    # TODO: Map pooled graph embeddings to regression predictions via a linear head.
    
    result=graph_embeddings@weight.T
    if bias is None:
        output=result
    else:
        output=result+bias
    return output

# Step 32 - generate_sbm_graph
def generate_sbm_graph(num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    # TODO: Sample one SBM graph with community labels and random node features.
    
    if seed is not None:
        torch.manual_seed(seed)
    
    node_features=torch.randn(num_nodes,feature_dim)

    node_labels=torch.zeros(num_nodes,dtype=torch.long)
    for c in range(num_classes):
        start=c*num_nodes//num_classes
        end=(c+1)*num_nodes//num_classes
        node_labels[start:end]=c

    src_list=[]
    dst_list=[]
    for i in range(num_nodes):
        for j in range(i+1,num_nodes):
            if node_labels[i]==node_labels[j]:
                prob=p_in
            else:
                prob=p_out

            if torch.rand(1).item()<prob:
                src_list.extend([i,j])
                dst_list.extend([j,i])
    edge_index=torch.tensor([src_list,dst_list],dtype=torch.long)

    return {
        'node_features':node_features,
        'edge_index':edge_index,
        'node_labels':node_labels,
        'num_nodes':num_nodes 
    }

# Step 33 - build_node_classification_dataset
def build_node_classification_dataset(num_graphs, num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    # TODO: Build a list of SBM graphs with consistent schema for node classification.
    dataset=[]
    for i in range(num_graphs):

        if seed is not None:
           current_seed=seed+i
        else:
           current_seed=None
    
        result=generate_sbm_graph(num_nodes,num_classes,p_in,p_out,feature_dim,seed=current_seed)
        dataset.append(result)
    
    return dataset

# Step 34 - generate_molecule_like_graph
def generate_molecule_like_graph(num_nodes, num_node_features, edge_prob=0.3, seed=0):
    # TODO: Synthesize one molecule-like graph with features, edges, and target...
    if seed is not None:
        torch.manual_seed(seed)
    x=torch.randn(num_nodes,num_node_features)
    src_list=[]
    dst_list=[]
    for i in range(num_nodes):
        for j in range(i+1,num_nodes):
            if torch.rand(1).item()<edge_prob:
                src_list.extend([i,j])
                dst_list.extend([j,i])
    edge_index=torch.tensor([src_list,dst_list],dtype=torch.long)
    deg=torch.bincount(torch.tensor(src_list,dtype=torch.long),minlength=num_nodes)
    mean_x=torch.mean(x,dim=1)
    N=deg*mean_x
    y=torch.mean(N.float())
    return {
        'x':x,
        'edge_index':edge_index,
        'y':y,
    }

# Step 35 - build_graph_regression_dataset (not yet solved)
# TODO: implement

# Step 36 - collate_graph_batch (not yet solved)
# TODO: implement

# Step 37 - cross_entropy_loss
import torch.nn.functional as F

def cross_entropy_loss(logits, targets):
    # TODO: Compute mean multi-class cross-entropy between logits and targets.
    log_probs=F.log_softmax(logits,dim=1)
    
    M=logits.shape[0]
    target_log_probs=log_probs[torch.arange(M),targets]

    return  -target_log_probs.mean()

# Step 38 - mse_loss
def mse_loss(predictions, targets):
    # TODO: Compute mean squared error between predictions and targets
    preds_flat=predictions.view(-1)
    targets_flat=targets.view(-1)
    diff=preds_flat-targets_flat
    result=torch.mean(diff**2)
    return  result

# Step 39 - accuracy_metric
def accuracy_metric(logits, targets):
    # TODO: Return the fraction of argmax(logits) predictions matching targets.
    preds=torch.argmax(logits,dim=1)
    correct_mask=(preds==targets)
    accuracy=correct_mask.float().mean()
         
    return accuracy.item()

# Step 40 - mae_metric
def mae_metric(predictions, targets):
    # TODO: Compute mean absolute error between predicted and target continuous values.
    loss=predictions.view(-1)-targets.view(-1)
    result=torch.mean(torch.abs(loss))
    return result.item()

# Step 41 - gnn_train_step
def gnn_train_step(params, batch, forward_fn, loss_fn, lr):
    # TODO: Run one SGD training step and update params in-place...
    predictions=forward_fn(params,batch)
    targets=batch['y']
    loss=loss_fn(predictions,targets)
    
    for param in params.values():
        if param.grad is not None:
            param.grad.zero_()


    loss.backward()
    with torch.no_grad():
       for param in params.values():
           param-=lr*param.grad
    
    return {
        'loss':loss.item(),
        'params':params
    }
    

    1

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

