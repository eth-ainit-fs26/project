"""
The MIT License

Copyright (c) 2020 Yeong-Dae Kwon
Copyright (c) 2026 Department of Computer Science, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.


THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .parameters import NODE_FEATURE_DIM

# For debugging
from IPython.core.debugger import set_trace

########################################
# ACTOR
########################################
class ACTOR(nn.Module):
    '''
    This class defines the ACTOR network for CVRP using grouped actors.
    The ACTOR consists of an encoder that maps the CVRP state to embeddings,
    and a next-node probability calculator that maps embeddings to action probabilities.
    '''
    def __init__(self, 
                 device: str,
                 EMBEDDING_DIM: int,
                 ENCODER_LAYER_NUM: int,
                 HEAD_NUM: int,
                 KEY_DIM: int,
                 FF_HIDDEN_DIM: int,
                 LOGIT_CLIPPING: float):
        super().__init__()

        # Set up hyperparameters; you do not need to worry about these
        self.device = device
        self.NODE_FEATURE_DIM = NODE_FEATURE_DIM
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.ENCODER_LAYER_NUM = ENCODER_LAYER_NUM
        self.HEAD_NUM = HEAD_NUM
        self.KEY_DIM = KEY_DIM
        self.FF_HIDDEN_DIM = FF_HIDDEN_DIM
        self.LOGIT_CLIPPING = LOGIT_CLIPPING

        # The actor consists of two sub-networks: encoder and next-node probability calculator
        # 1. Set up encoder network which maps CVRP state to embeddings
        self.encoder = Encoder(
            NODE_FEATURE_DIM = self.NODE_FEATURE_DIM,
            EMBEDDING_DIM = self.EMBEDDING_DIM, 
            ENCODER_LAYER_NUM = self.ENCODER_LAYER_NUM, 
            HEAD_NUM = self.HEAD_NUM, 
            KEY_DIM = self.KEY_DIM,
            FF_HIDDEN_DIM = self.FF_HIDDEN_DIM
            ).to(self.device)
        # 2. Set up next-node probability calculator which maps embeddings to probabilities 
        # over next nodes as actions to take
        self.node_prob_calculator = Next_Node_Probability_Calculator_for_Group(
                                        EMBEDDING_DIM = self.EMBEDDING_DIM,
                                        HEAD_NUM = self.HEAD_NUM,
                                        KEY_DIM = self.KEY_DIM,
                                        LOGIT_CLIPPING = self.LOGIT_CLIPPING
                                    ).to(self.device)

        self.batch_s = None         # batch size
        self.encoded_nodes = None   # node embeddings from encoder
        self.encoded_graph = None   # graph embedding (mean of node embeddings)

    def reset(self, group_state):
        '''Reset the actor for a new (initial) group state.'''

        self.batch_s = group_state.data.size(0) # batch size
        # node embeddings; shape = (batch, problem+1, EMBEDDING_DIM)
        self.encoded_nodes = self.encoder(
            group_state.data.to(self.device),
            group_state.cost_matrix.to(self.device)
        )
        # graph embedding; shape = (batch, 1, EMBEDDING_DIM)
        self.encoded_graph = self.encoded_nodes.mean(dim=1, keepdim=True)
        # reset the node probability calculator with the new node embeddings
        self.node_prob_calculator.reset(self.encoded_nodes)

    def pick_nodes_for_each_group(self, encoded_nodes, node_index_to_pick):
        '''Retrieve node embeddings for each tour based on the provided node indices.
        Parameters:
            encoded_nodes: FloatTensor of shape (batch, problem, EMBEDDING_DIM) containing node embeddings
            node_index_to_pick: LongTensor of shape (batch, group) containing indices of nodes to pick
        Returns:
            picked_nodes: FloatTensor of shape (batch, group, EMBEDDING_DIM) containing picked node embeddings
        '''

        gathering_index = node_index_to_pick[:, :, None].expand(-1, -1, self.EMBEDDING_DIM)
        # shape = (batch, group, EMBEDDING_DIM)
        
        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
        # shape = (batch, group, EMBEDDING_DIM)
        
        return picked_nodes

    def get_action_probabilities(self, group_state):
        '''Calculate the action probabilities for the current group state.
        Parameters:
            group_state: GROUP_STATE instance representing the current state of the environment
        Returns:
            item_select_probabilities: FloatTensor of shape (batch, group, problem+1) containing action probabilities
        '''

        # 1. Retrieve the embeddings of the last nodes for each tour
        encoded_LAST_NODES = self.pick_nodes_for_each_group(self.encoded_nodes, 
                                                       group_state.current_node.to(self.device))
        # shape = (batch, group, EMBEDDING_DIM)

        # 2. Retrieve the remaining capacity for each tour
        remaining_capacity = group_state.remaining_capacity[:, :, None].to(self.device)
        rem_cap_scaled = remaining_capacity / group_state.vehicle_capacity
        # shape = (batch, group, 1)

        # 3. Calculate action probabilities using the node probability calculator
        # For each tour, the next node probabilities are calculated based on the embedding of
        # the graph, embeddings of the last visited node, the remaining capacity, and the ninf_mask
        # which masks out invalid actions
        item_select_probabilities = self.node_prob_calculator(self.encoded_graph, 
                                                              encoded_LAST_NODES,
                                                              rem_cap_scaled, 
                                                              ninf_mask=group_state.ninf_mask.to(self.device))
        # shape = (batch, group, problem+1)

        return item_select_probabilities
        


########################################
# ACTOR_SUB_NN : ENCODER
########################################

class Encoder(nn.Module):
    def __init__(self, 
                 NODE_FEATURE_DIM: int,
                 EMBEDDING_DIM: int, 
                 ENCODER_LAYER_NUM: int,
                 HEAD_NUM: int,
                 KEY_DIM: int, 
                 FF_HIDDEN_DIM: int):
        super().__init__()
        self.NODE_FEATURE_DIM = NODE_FEATURE_DIM
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.ENCODER_LAYER_NUM = ENCODER_LAYER_NUM
        self.HEAD_NUM = HEAD_NUM
        self.KEY_DIM = KEY_DIM
        self.FF_HIDDEN_DIM = FF_HIDDEN_DIM

        self.embedding_depot = nn.Linear(NODE_FEATURE_DIM, EMBEDDING_DIM)
        self.embedding_node = nn.Linear(NODE_FEATURE_DIM, EMBEDDING_DIM)
        self.layers = nn.ModuleList([Encoder_Layer(EMBEDDING_DIM=EMBEDDING_DIM, 
                                                   HEAD_NUM=HEAD_NUM, 
                                                   KEY_DIM=KEY_DIM, 
                                                   FF_HIDDEN_DIM=FF_HIDDEN_DIM) 
                                     for _ in range(ENCODER_LAYER_NUM)])

    def forward(self, data, cost_matrix):
        # data.shape = (batch, problem+1, NODE_FEATURE_DIM) 
        # cost_matrix.shape = (batch, problem+1, problem+1)
        depot_features = data[:, [0], :] # shape = (batch, 1, NODE_FEATURE_DIM)
        node_features = data[:, 1:, :]   # shape = (batch, problem, NODE_FEATURE_DIM)

        embedded_depot = self.embedding_depot(depot_features)
        # shape = (batch, 1, EMBEDDING_DIM)
        embedded_node = self.embedding_node(node_features)
        # shape = (batch, problem, EMBEDDING_DIM)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape = (batch, problem+1, EMBEDDING_DIM)

        for layer in self.layers:
            out = layer(out, cost_matrix)

        return out


class Encoder_Layer(nn.Module):
    def __init__(self, EMBEDDING_DIM: int, HEAD_NUM: int, KEY_DIM: int, FF_HIDDEN_DIM: int):
        super().__init__()
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.HEAD_NUM = HEAD_NUM
        self.KEY_DIM = KEY_DIM
        self.FF_HIDDEN_DIM = FF_HIDDEN_DIM

        self.Wq = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wk = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wv = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.multi_head_combine = nn.Linear(HEAD_NUM * KEY_DIM, EMBEDDING_DIM)

        ### The 2-layer MLP to map scalar cost to Attention Head Biases
        self.bias_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, HEAD_NUM)
        )

        self.addAndNormalization1 = Add_And_Normalization_Module(EMBEDDING_DIM)
        self.feedForward = Feed_Forward_Module(EMBEDDING_DIM, FF_HIDDEN_DIM)
        self.addAndNormalization2 = Add_And_Normalization_Module(EMBEDDING_DIM)

    def forward(self, input1, cost_matrix):
        # input1.shape = (batch, problem+1, EMBEDDING_DIM)
        # cost_matrix.shape = (batch, problem+1, problem+1)

        q = reshape_by_heads(self.Wq(input1), head_num=self.HEAD_NUM)
        k = reshape_by_heads(self.Wk(input1), head_num=self.HEAD_NUM)
        v = reshape_by_heads(self.Wv(input1), head_num=self.HEAD_NUM)
        # q shape = (batch, HEAD_NUM, problem+1, KEY_DIM)

        # Compute attention bias from cost matrix
        # 1. Expand cost matrix to (batch, problem+1, problem+1, 1)
        cost_matrix_expanded = cost_matrix.unsqueeze(-1)
        # 2. Pass through MLP to get (batch, problem+1, problem+1, HEAD_NUM)
        bias = self.bias_mlp(cost_matrix_expanded)
        # 3. Permute to match attention logit shape: (batch, HEAD_NUM, problem+1, problem+1)
        attn_bias = bias.permute(0, 3, 1, 2)
        # 4. Pass the bias into the attention calculation
        out_concat = multi_head_attention(q, k, v, attn_bias=attn_bias)
        # shape = (batch, problem+1, HEAD_NUM*KEY_DIM)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape = (batch, problem+1, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3


########################################
# ACTOR_SUB_NN : Next_Node_Probability_Calculator
########################################

class Next_Node_Probability_Calculator_for_Group(nn.Module):
    def __init__(self,
                 EMBEDDING_DIM: int,
                 HEAD_NUM: int,
                 KEY_DIM: int,
                 LOGIT_CLIPPING: float):
        super().__init__()
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.HEAD_NUM = HEAD_NUM
        self.KEY_DIM = KEY_DIM
        self.LOGIT_CLIPPING = LOGIT_CLIPPING

        self.Wq = nn.Linear(2*EMBEDDING_DIM+1, HEAD_NUM * KEY_DIM, bias=False)
        self.Wk = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)
        self.Wv = nn.Linear(EMBEDDING_DIM, HEAD_NUM * KEY_DIM, bias=False)

        self.multi_head_combine = nn.Linear(HEAD_NUM * KEY_DIM, EMBEDDING_DIM)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention

    def reset(self, encoded_nodes):
        # encoded_nodes.shape = (batch, problem+1, EMBEDDING_DIM)

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=self.HEAD_NUM)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=self.HEAD_NUM)
        # shape = (batch, HEAD_NUM, problem+1, KEY_DIM)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape = (batch, EMBEDDING_DIM, problem+1)

    def forward(self, input1, input2, remaining_capacity, ninf_mask=None):
        # input1.shape = (batch, 1, EMBEDDING_DIM) # graph embedding
        # input2.shape = (batch, group, EMBEDDING_DIM) # last node embeddings for each tour
        # remaining_capacity.shape = (batch, group, 1) # remaining capacity for each tour
        # ninf_mask.shape = (batch, group, problem+1) # -inf mask for invalid actions

        group_s = input2.size(1)

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((input1.expand(-1, group_s, -1), input2, remaining_capacity), dim=2)
        # shape = (batch, group, 2*EMBEDDING_DIM+1)

        q = reshape_by_heads(self.Wq(input_cat), head_num=self.HEAD_NUM)
        # shape = (batch, HEAD_NUM, group, KEY_DIM)

        out_concat = multi_head_attention(q, self.k, self.v, ninf_mask=ninf_mask)
        # shape = (batch, n, HEAD_NUM*KEY_DIM), n=1 or group

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape = (batch, n, EMBEDDING_DIM), n=1 or group

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape = (batch, n, problem+1)

        score_scaled = score / np.sqrt(self.EMBEDDING_DIM)
        # shape = (batch_s, group, problem+1)

        score_clipped = self.LOGIT_CLIPPING * torch.tanh(score_scaled)

        if ninf_mask is None:
            score_masked = score_clipped
        else:
            score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape = (batch, group, problem+1)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape = (batch, C, head_num*key_dim)

    batch_s = qkv.size(0)
    C = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, C, head_num, -1)
    # shape = (batch, C, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape = (batch, head_num, C, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, ninf_mask=None, attn_bias=None):
    # q shape = (batch, head_num, n, key_dim)   : n can be either 1 or group
    # k,v shape = (batch, head_num, problem, key_dim)
    # ninf_mask.shape = (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    problem_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / np.sqrt(key_dim)
    # shape = (batch, head_num, n, problem)

    # Inject the geometric bias directly into the raw attention logits
    if attn_bias is not None:
        score_scaled = score_scaled + attn_bias

    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, None, :, :].expand(batch_s, head_num, n, problem_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape = (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape = (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape = (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape = (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, EMBEDDING_DIM: int):
        super().__init__()
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.norm_by_EMB = nn.BatchNorm1d(EMBEDDING_DIM, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape = (batch, problem, EMBEDDING_DIM)
        batch_s = input1.size(0)
        problem_s = input1.size(1)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, self.EMBEDDING_DIM))

        return normalized.reshape(batch_s, problem_s, self.EMBEDDING_DIM)


class Feed_Forward_Module(nn.Module):
    def __init__(self, EMBEDDING_DIM: int, FF_HIDDEN_DIM: int):
        super().__init__()

        self.W1 = nn.Linear(EMBEDDING_DIM, FF_HIDDEN_DIM)
        self.W2 = nn.Linear(FF_HIDDEN_DIM, EMBEDDING_DIM)

    def forward(self, input1):
        # input.shape = (batch, problem, EMBEDDING_DIM)

        return self.W2(F.relu(self.W1(input1)))
