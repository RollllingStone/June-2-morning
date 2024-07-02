import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer
import Utils
from torch_geometric.nn import GCNConv


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    #upper triangle matrix so that future position are masked out
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)

    # replicates the mask across all sequences in the batch to ensure each sequence in the batch uses the same masking pattern.
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        self.linear_demand = nn.Linear(1, d_model)

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cpu'))

        # event type embedding
        self.num_types = num_types
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)

        # self.demand_emb = nn.Linear(1, d_model)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def demand_enc(self, demand, non_pad_mask):
        # demand shape: [batch_size, seq_len]
        # Unsqueeze the last dimension to make x of shape [batch_size, seq_len, 1]
        demand = demand.unsqueeze(-1)
        # Output shape: [batch_size, seq_len, d_model]
        output = self.linear_demand(demand)
        return output * non_pad_mask


    def forward(self, event_type, event_time, event_demand, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        #改这一个就可以
        # enc_output = self.event_emb(event_type)  
        demand_enc = self.demand_enc(event_demand, non_pad_mask)
        # enc_output += demand_enc

        # for enc_layer in self.layer_stack:
        #     #在这加的
        #     enc_output += tem_enc
        #     enc_output, _ = enc_layer(
        #         enc_output,
        #         non_pad_mask=non_pad_mask,
        #         slf_attn_mask=slf_attn_mask)


        enc_output = self.event_emb(event_type) + tem_enc + demand_enc

        # Process each layer in the stack
        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        return enc_output


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, d_model):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, d_model)

    def forward(self, x, edge_index):
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        return x


# def process_graph_sequences(model, batch_graph_sequences, non_pad_mask):
#     batch_outputs = []
#     for single_store_graphs in batch_graph_sequences:
#         single_store_graphs = []
#         for graph in single_store_graphs:
#             # if graph.num_nodes == 0:

#             out = model(graph.x, graph.edge_index)
#             # Retrieve the node feature for the root node, by contrustruction应为subgraph第一个element
#             root_node_feature = out[0]
#             single_store_graphs.append(root_node_feature)
#         batch_outputs.append(single_store_graphs)

#     batch_outputs = np.array(batch_outputs)
#     print('batch_outputs', batch_outputs.shape)
#     return torch.tensor(batch_outputs, dtype=torch.float32)

def process_graph_sequences(model, batch_graph_sequences, non_pad_mask, d_model):

    num_batches = len(batch_graph_sequences)  # Total number of batches
    graphs_per_batch = len(batch_graph_sequences[0])  # Number of graphs per batch (assuming this is consistent)
    feature_size = d_model  # Assuming each root node feature vector is of length 32

    # Preallocate the NumPy array
    batch_outputs = np.zeros((num_batches, graphs_per_batch, feature_size))

    # Processing loop
    # for i, single_store_graphs in enumerate(batch_graph_sequences):
    #     for j, graph in enumerate(single_store_graphs):
    #         if graph.num_nodes == 0:
    #             continue  
    #         # print("Node features size:", graph.x.size())
    #         # print("Edge index size:", graph.edge_index.size())

    #         out = model(graph.x, graph.edge_index)
    #         root_node_feature = out[0].detach().cpu().numpy()  # Ensure tensor is moved to CPU and converted to NumPy
    #         batch_outputs[i, j] = root_node_feature


    for i, single_store_graphs in enumerate(batch_graph_sequences):
        for j, graph in enumerate(single_store_graphs):
            # print(single_store_graphs[0].x, 'graph 1')
            # print(single_store_graphs[1].x,'graph 2')
            # print(single_store_graphs[2].x,'graph 3')

            if graph.num_nodes == 0:
                print(f"Graph at batch {i}, position {j} is empty.")
                continue
            
            out = model(graph.x, graph.edge_index)

            root_node_feature = out[0].detach().cpu().numpy()  # Assuming the root node is the first node
            
            batch_outputs[i, j] = root_node_feature

    return torch.tensor(batch_outputs, dtype=torch.float32)


# def process_graph_sequences(model, graph_sequences, node_index_list, non_pad_mask):
#     outputs = []
#     for batch in graph_sequences:  # Assuming each batch is already a Batch object
#         # node_index = node_index_list[i]
#         out = model(batch.x, batch.edge_index, batch.batch)
#         # Retrieve the node feature for the specified node index
#         # node_feature = out[node_index]
#         node_feature = out[0]
#         outputs.append(node_feature)
#     return outputs



class MaskedLayerNorm(nn.Module):
    def __init__(self, model_dim):
        super(MaskedLayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(model_dim))
        self.shift = nn.Parameter(torch.zeros(model_dim))

    def forward(self, x, mask):
        # Ensure mask is boolean and the same shape as x
        mask = mask.unsqueeze(-1).expand_as(x)  # Expand mask to match x dimensions

        # Masking the input and compute mean and std only on non-masked (non-padded) elements
        masked_x = torch.where(mask, x, torch.tensor(0.0, device=x.device))
        mean = masked_x.sum(dim=(1, 2)) / mask.sum(dim=(1, 2))
        variance = ((masked_x - mean.unsqueeze(1).unsqueeze(2)) ** 2).sum(dim=(1, 2)) / mask.sum(dim=(1, 2))
        std = torch.sqrt(variance + 1e-5)

        # Normalize
        x = (masked_x - mean.unsqueeze(1).unsqueeze(2)) / std.unsqueeze(1).unsqueeze(2)
        return x * self.scale + self.shift

# Example usage
batch_size = 10
seq_len = 20
model_dim = 64
x = torch.randn(batch_size, seq_len, model_dim)  # Example tensor
mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.bool)  # Example boolean mask

# Create the layer
layer_norm = MaskedLayerNorm(model_dim)
normalized_output = layer_norm(x, mask)


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, num_node_features, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):

        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )
        #gcn output dim 其实不一定是d_model 无所谓

        # print(num_node_features,'num_node_features')
        self.gcn = GCN(num_node_features, d_model)

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)
        # self.linear_lamda = nn.Linear(d_model, num_types)

        # self.linear_alpha = nn.Linear(d_model, 1)
        # self.linear_alpha = nn.Linear(d_model, 1)

        self.d_model = d_model


        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))


        # Parameter v (vector) that should match the size of the hidden states
        # self.v_lamda = nn.Parameter(torch.randn(d_model))
        # self.v_lambda = nn.Parameter(torch.randn(d_model, num_types))
        self.v_alpha = nn.Parameter(0.01 * torch.abs(torch.randn(2 * d_model, 1)))
        self.v_beta = nn.Parameter(0.01 * torch.abs(torch.randn(2 * d_model, 1)))
        
        # # Parameter b_i (scalar or vector)   
        #为了 predict time的
        # self.b_lamda = nn.Parameter(torch.randn(num_stores))
        # self.b_alpha = nn.Parameter(torch.randn(num_stores))
        # self.b_beta = nn.Parameter(torch.randn(num_stores))


        self.b_alpha = nn.Parameter(torch.full((1,), 1.0))
        self.b_beta = nn.Parameter(torch.full((1,), 1.0))
        # self.b_alpha = nn.Parameter(torch.randn(1))
        # self.b_beta = nn.Parameter(torch.randn(1))


        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)


        #v x s + b
        self.w = nn.Parameter(torch.randn(2 * d_model, 1))
        #unique gpt也给你写过
        # self.bias = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(3 * abs(torch.randn(3282)))



        # prediction of next time stamp
        self.time_predictor = Predictor(d_model, 1)

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)


        self.bias_alpha = nn.Embedding(3282, 1)
        self.bias_alpha.weight.data.fill_(3.0)
        self.bias_beta = nn.Embedding(3282, 1)
        self.bias_beta.weight.data.fill_(3.0)


        self.layer_norm1 = MaskedLayerNorm(model_dim)
        self.layer_norm2 = MaskedLayerNorm(model_dim)






    def forward(self, event_type, event_time, event_demand, batch_placekey_list, batch_place_graph_seq):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: [batch_size, store_sequence_len]
               event_time: [batch_size, store_sequence_len]
               event_demand: batch* seq_len


        Output: enc_output: [batch_size, store_sequence_len, model_dim]
                type_prediction:(not normalized) [batch_size, store_sequence_len, num_classes]
                time_prediction: [batch_size, store_sequence_len]
        """
        # D 凑一个M dim 出来  X = (Z + (UY) + D)

        batch_placekey_list = torch.tensor(batch_placekey_list, dtype=torch.long)

        non_pad_mask = get_non_pad_mask(event_type)

        enc_output = self.encoder(event_type, event_time, event_demand, non_pad_mask)
        enc_output = self.rnn(enc_output, non_pad_mask)

        #node_index_list是要subgraph的root node在subgraph里面的index

        graph_output = process_graph_sequences(self.gcn, batch_place_graph_seq,non_pad_mask, self.d_model)


        # print("enc_output shape:", enc_output.shape)
        print("graph_output shape:", graph_output.shape)
        # print(graph_output, 'graph_output')

        # graph_output = self.ffn(graph_output)

        # Concatenating along the last dimension


        print(norm_enc_output.shape, 'enc_output')
        print(norm_graph_out.shape,'graph_output')
        s_representation = torch.cat((norm_enc_output, norm_graph_out), dim=2)
        # lamda = softplus(self.w * s_representation + self.bias)
        inside_softplus_lamda = torch.matmul(s_representation, self.w) 
        inside_softplus_lamda += self.bias(batch_placekey_list).unsqueeze(1)
        lamda = Utils.softplus(inside_softplus_lamda, self.beta)



        normalized_output = layer_norm(x, mask)



        time_prediction = self.time_predictor(enc_output, non_pad_mask)

        type_prediction = self.type_predictor(enc_output, non_pad_mask)


        # s_representation = self.norm(s_representation)

        # print('s_representation', s_representation)




        # # =========    time prediction    ====================
        # all_hid = torch.matmul(enc_output, self.v_lamda)
        # all_hid += self.b_lamda[store_idx].unsqueeze(1).unsqueeze(2)
        # all_lambda = softplus(all_hid, model.beta)
        # type_lambda = torch.sum(all_lambda * non_pad_mask, dim=2)
        # #type_lamda shape: [batch_size, seq_len]


        # =========    Demand prediction    ====================
        hid_v_alpha = torch.matmul(s_representation, self.v_alpha)
        #individual b应该是可以做到的
        # hid_v_alpha += self.b_alpha[store_idx].unsqueeze(1).unsqueeze(2)
        hid_v_alpha += self.bias_alpha(batch_placekey_list).unsqueeze(1)
        # hid_v_alpha += self.b_alpha.unsqueeze(1).unsqueeze(2)
        all_alpha = Utils.softplus(hid_v_alpha, self.beta)
        #这个是算total lamda rate, 我们一个type  不影响
        type_alpha = torch.sum(all_alpha * non_pad_mask, dim=2)
        #type_alpha shape: [batch_size, seq_len]

        hid_v_beta = torch.matmul(s_representation, self.v_beta)
        # hid_v_beta += self.b_beta[store_idx].unsqueeze(1).unsqueeze(2)
        # hid_v_beta += self.b_beta.unsqueeze(1).unsqueeze(2)
        hid_v_beta += self.bias_beta(batch_placekey_list).unsqueeze(1)
        all_beta = Utils.softplus(hid_v_beta, self.beta)
        #这个是算total lamda rate, 我们一个type  不影响
        # type_beta = torch.sum(all_beta * non_pad_mask, dim=2)

        epsilon = 1e-8

        demand_prediction = all_alpha / (all_beta + epsilon)
        #demand_prediction shape : [batch_size, seq_len]
        demand_prediction *= non_pad_mask

        #现在其实不影响train  只是predict time会受影响


        print("hid_v_alpha:", hid_v_alpha.detach().cpu().numpy())
        print("all alpha:", all_alpha.detach().cpu().numpy())
        print("type alpha shape:", type_alpha.shape)
        print("type alpha:", type_alpha.detach().cpu().numpy())



        print("hid_v_beta:", hid_v_beta.detach().cpu().numpy())
        print("Softplus beta:", all_beta.detach().cpu().numpy())


        # return enc_output, (type_prediction, time_prediction, demand_prediction), (self.b_alpha, self.b_beta)
        print(time_prediction.tolist(), 'time prediction here')
        print(demand_prediction.tolist(), 'demand prediction here')
        print(type_prediction.tolist(), 'type prediction here')

        return s_representation, (type_prediction, time_prediction, demand_prediction)
        # return enc_output, (type_prediction, time_prediction, demand_prediction)




