import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
import transformer.Constants as Constants
import Utils
import pandas as pd
from torch_geometric.data import Data
# from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm
from torch_geometric.utils import subgraph
import numpy as np
import torch
import torch.utils.data
from collections import defaultdict

from transformer import Constants
import ast


#  ===================================   dataset.py     =================================== 


# class EventData(torch.utils.data.Dataset):
#     """ Event stream dataset. """

#     def __init__(self, data):
      
#         self.placekey_list = list(range(len(list(data['placekey']))))
#         all_store_events = list(data['event_sequence'])

#         all_store_events = [ast.literal_eval(event_string) for event_string in data['event_sequence']]

#         self.time = [[single_event['time'] for single_event in single_store_events] for single_store_events in all_store_events]
#         self.time_gap = [[single_event['time_since_last_event'] for single_event in single_store_events] for single_store_events in all_store_events]
#         # plus 1 since there could be event type 0, but we use 0 as padding
#         self.demand_marker = [[single_event['actual_demand'] + 0.00001 for single_event in single_store_events] for single_store_events in all_store_events]
#         # plus 1 since there could be event type 0, but we use 0 as padding
#         self.event_type = [[1 for single_event in single_store_events] for single_store_events in all_store_events]

#         self.length = len(all_store_events)

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         """ Each returned element is a list, which represents an event stream """
#         return self.time[idx], self.time_gap[idx], self.event_type[idx], self.demand_marker[idx], self.placekey_list[idx]


def min_max_scale(demand, min_demand, max_demand):
    # Avoid division by zero if all values are the same
    if max_demand == min_demand:
        return 0 if max_demand == 0 else 1
    return (demand - min_demand) / (max_demand - min_demand)


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
      
        self.placekey_list = list(range(len(list(data['placekey']))))
        all_store_events = list(data['event_sequence'])

        all_store_events = [ast.literal_eval(event_string) for event_string in data['event_sequence']]

        self.time = [[single_event['time'] for single_event in single_store_events] for single_store_events in all_store_events]
        self.time_gap = [[single_event['time_since_last_event'] for single_event in single_store_events] for single_store_events in all_store_events]
        # plus 1 since there could be event type 0, but we use 0 as padding
        # self.demand_marker = [[single_event['actual_demand'] + 0.00001 for single_event in single_store_events] for single_store_events in all_store_events]
        self.demand_marker = [[single_event['actual_demand'] for single_event in single_store_events] for single_store_events in all_store_events]

        max_demand = [ max(single_store_demand) for single_store_demand in self.demand_marker]
        min_demand = [ min(single_store_demand) for single_store_demand in self.demand_marker]

        self.length = len(all_store_events)


        self.demand_marker_scaled = [[(demand - min_demand[idx]) / (max_demand[idx] - min_demand[idx] if max_demand[idx] != min_demand[idx] else 1) # Avoid division by zero
        for demand in store_demands] for idx, store_demands in enumerate(self.demand_marker)]
        # print(self.demand_marker_scaled, 'demand marker scaled')

        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[1 for single_event in single_store_events] for single_store_events in all_store_events]

        # self.length = len(all_store_events)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx], self.demand_marker_scaled[idx],self.demand_marker[idx], self.placekey_list[idx]


def pad_time(all_store_events):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(single_store_events) for single_store_events in all_store_events)

    # single sided padding
    batch_seq = np.array([
        single_store_events + [Constants.PAD] * (max_len - len(single_store_events))
        for single_store_events in all_store_events])


    # max_len = max(len(inst) for inst in insts)

    # batch_seq = np.array([
    #     inst + [Constants.PAD] * (max_len - len(inst))
    #     for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


# def pad_placekey(insts):
#     """ Pad the instance to the max seq length in batch. """

#     max_len = max(len(inst) for inst in insts)

#     batch_seq = np.array([
#         inst + [Constants.PAD] * (max_len - len(inst))
#         for inst in insts])

#     return batch


def pad_demand_mark(all_store_events):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(single_store_events) for single_store_events in all_store_events)

    # single sided padding
    batch_seq = np.array([
        single_store_events + [Constants.PAD] * (max_len - len(single_store_events))
        for single_store_events in all_store_events])

    return torch.tensor(batch_seq, dtype=torch.float32)

def create_empty_graph_pyg():
    # Create an empty graph with no nodes and no edges
    edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
    x = torch.empty((0, 0))  # No nodes and hence no features
    
    empty_graph = Data(x=x, edge_index=edge_index)
    return empty_graph

def pad_graph(all_place_graph_seq):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(single_store_graphs) for single_store_graphs in all_place_graph_seq)

    batch_seq = [
        single_store_graphs + [create_empty_graph_pyg()] * (max_len - len(single_store_graphs))
        for single_store_graphs in all_place_graph_seq]
      
    return batch_seq



def create_ego_centric_graph(data, placekey_index):
    # Get the neighbors + the node itself
    neighbors = torch.cat([ torch.tensor([placekey_index], device=data.edge_index.device),
                            data.edge_index[0][data.edge_index[1] == placekey_index],
                           data.edge_index[1][data.edge_index[0] == placekey_index]])

    neighbors = torch.unique(neighbors)  # Remove duplicates
    #应该对应着node_indics  原图 index合集  可以去对应小图的index
    neighbors = neighbors.tolist()


    # Extract the subgraph
    # sub_data = subgraph(neighbors, data.edge_index, num_nodes=data.num_nodes, edge_attr=data.edge_attr, relabel_nodes = False)
    # new_edge_index, edge_mask = subgraph(neighbors, data.edge_index, num_nodes=data.num_nodes, relabel_nodes=False)
    new_edge_index, edge_mask = subgraph(neighbors, data.edge_index, num_nodes=data.num_nodes, relabel_nodes=True)
    sub_data = Data(x=data.x[neighbors].clone(), edge_index=new_edge_index)


    # if 'edge_attr' in data:
        # sub_data.edge_attr = data.edge_attr[edge_mask]  # Only if edge attributes exist and are relevant

    # print(type(sub_data), 'sub_data')
    #neighbors是 原图 index合集  可以用.index()去对应小图的index
    return sub_data, neighbors

def add_demand_features(sub_graph, spend_df, placekey_index_list, placekey_index_mapping_dict):
    #placekey_index_list就是neighbors_list  对应大图中每个node的序号
    index_placekey_mapping_dict = placekey_index_mapping_dict

    # print(len(spend_df['spend'].tolist()[0]),'graph365')

    # print(spend_df,'spend_df')
    #place_index_list是original图的index list
    for index in placekey_index_list:
      #这样的index → 文字  {0:'sfesefsdsdfds', 1: 'ssdfsef3',...}
      placekey = index_placekey_mapping_dict[index]
      # print(placekey,'placekey')

      # spend_data = spend_df.loc[spend_df['PLACEKEY'] == placekey, 'spend'].apply(ast.literal_eval)
      spend_data = spend_df.loc[spend_df['PLACEKEY'] == placekey, 'spend']

      # print(spend_data,'spend here')

      spend_data = np.array(spend_data.tolist())
      # print(spend_data,'spend data')
      spend_tensor = torch.tensor(spend_data, dtype=torch.float)
      spend_tensor = spend_tensor.flatten()

      # existing_features = sub_graph.x[index]
      # print(existing_features.shape, 'shape')
      # print(spend_tensor.shape, 'shape')

      num_nodes = sub_graph.x.size(0)  # Total number of nodes in the subgraph
      feature_length = sub_graph.x.size(1)  # Number of features per node

      # print(num_nodes,'num_nodes')
      # print(feature_length,'feature_length')

      # new_features = torch.cat([existing_features, spend_tensor], dim=0) 

      index_in_subgraph = placekey_index_list.index(index)

      sub_graph.x[index_in_subgraph, -30:] = spend_tensor

      # sub_graph.x[index] = new_features

    return sub_graph

class CollateFunction:
    def __init__(self, G, index_neighbor_mapping_dict, graph_365,placekey_index_mapping_dict):
        self.placekey_index_mapping_dict = placekey_index_mapping_dict
        self.G = G  # Store G as an instance variable
        self.index_neighbor_mapping_dict = index_neighbor_mapping_dict
        self.graph_365 = graph_365

    def __call__(self, all_store_events):
        G = self.G
        index_neighbor_mapping_dict = self.index_neighbor_mapping_dict
        graph_365 = self.graph_365
        placekey_index_mapping_dict = self.placekey_index_mapping_dict

        time, time_gap, event_type, event_demand, event_demand_orignal, batch_placekey_list = [list(x) for x in zip(*all_store_events)]



        # def check_graph_uniqueness(graphs):
        #     unique_check = set()
        #     for i, graph in enumerate(graphs):
        #         feature_snapshot = tuple(graph.x.flatten().tolist())
        #         if feature_snapshot in unique_check:
        #             print(f"Duplicate found in graph {i}")
        #         else:
        #             unique_check.add(feature_snapshot)

        all_place_graph_seq = []

        for iter_index in range(len(batch_placekey_list)):
            #i是placekey在大图里对应的序号
            i = batch_placekey_list[iter_index]
            # egocentric_graph, neighbors_orig_graph_list = create_ego_centric_graph(G, i)
            #time sequence for current placekey
            time_seq = time[iter_index]

            graph_sequence_current_placekey =[]

            #这样的index → 文字  {0:'sfesefsdsdfds', 1: 'ssdfsef3',...}
            # placekey = placekey_index_mapping_dict[i]
            for t in time_seq:
                #whole graph dataframe at time t
                egocentric_graph, neighbors_orig_graph_list = create_ego_centric_graph(G, i)
                graph_dataframe_t = graph_365[t]
                # placekey_index_list = index_neighbor_mapping_dict[i]
                # new_graph = add_demand_features(egocentric_graph, graph_dataframe_t, placekey_index_list,placekey_index_mapping_dict)
                new_graph = add_demand_features(egocentric_graph, graph_dataframe_t, neighbors_orig_graph_list ,placekey_index_mapping_dict)
                graph_sequence_current_placekey.append(new_graph)

            # check_graph_uniqueness(graph_sequence_current_placekey)

            all_place_graph_seq.append(graph_sequence_current_placekey)

        time = pad_time(time)
        time_gap = pad_time(time_gap)
        event_demand = pad_demand_mark(event_demand)
        event_demand_orignal = pad_demand_mark(event_demand_orignal)
        event_type = pad_type(event_type)
        batch_placekey_list = batch_placekey_list
        all_place_graph_seq = pad_graph(all_place_graph_seq)

        return time, time_gap, event_type, event_demand, event_demand_orignal, batch_placekey_list, all_place_graph_seq


#  ===================================   Main.py     =================================== 

# def create_dataframe_day(t, dataframe):
#     start_idx = t - 30
#     end_idx = t

#     new_dataframe = pd.DataFrame()
#     new_dataframe['spend'] = dataframe['spend'].apply(ast.literal_eval)
#     print('finished first')
#     new_dataframe['spend'] = new_dataframe['spend'].apply(lambda x: x[start_idx:end_idx])
#     print('finished second')


#     new_dataframe['PLACEKEY'] = dataframe['PLACKEY']

#     return new_dataframe

def create_dataframe_day(t, dataframe):
    # Filter the DataFrame for the specific 't'
    filtered_df = dataframe[dataframe['time'] == t].reset_index(drop=True)

    # Process 'spend' data: parse strings and slice lists
    new_dataframe = pd.DataFrame()
    new_dataframe['spend'] = filtered_df['spend'].apply(ast.literal_eval)
    new_dataframe['PLACEKEY'] = filtered_df['PLACEKEY']
    # new_dataframe['time'] = t

    return new_dataframe

def create_index_neighbor_mapping(dist_matrix):
    index_neighbor_mapping_dict = {}
    num_nodes = dist_matrix.shape[0]
    
    for i in range(num_nodes):
        # Find indices of non-zero elements in the row
        neighbors = np.where(dist_matrix[i] != 0)[0]
        # Include itself in the neighbors list
        neighbors = np.append(neighbors, i)
        index_neighbor_mapping_dict[i] = np.unique(neighbors).tolist()  # Convert to list and remove duplicates if any
    
    return index_neighbor_mapping_dict

def prepare_encoder(data):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoder.fit(data.reshape(-1, 1))
    return encoder

#poi_df是non_sparse那个  
def create_G(dist_matrix, poi_df):
    num_nodes = dist_matrix.shape[0]

    additional_feature_space=30

    # Instantiate and fit the encoders
    # top_cat_encoder = prepare_encoder(poi_df['TOP_CATEGORY'].unique())
    sub_cat_encoder = prepare_encoder(poi_df['SUB_CATEGORY'].unique())

    # One-hot encode categories
    # top_category = top_cat_encoder.transform(poi_df['TOP_CATEGORY'].values.reshape(-1, 1))
    sub_category = sub_cat_encoder.transform(poi_df['SUB_CATEGORY'].values.reshape(-1, 1))


    # Convert the distance matrix to edge_index and edge_attr
    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if dist_matrix[i][j] != 0:  # Assuming non-zero value indicates an edge
                edge_index.append([i, j])
                edge_attr.append(dist_matrix[i][j])  # Edge weight

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Combine all node features into a single matrix
    # x = torch.tensor(np.concatenate((top_category, sub_category), axis=1), dtype=torch.float)
    x = torch.tensor(sub_category, dtype=torch.float)

    x = torch.cat([x, torch.zeros((num_nodes, additional_feature_space))], dim=1)  # Add space for future features


    # Create the PyTorch Geometric data object
    G = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    num_node_features = G.x.size(1)


    return G, num_node_features


def prepare_dataloader(opt, G, index_neighbor_mapping_dict,graph_365):
    """ Load data and prepare dataloader. """


    train_data = pd.read_csv('./train_only.csv')
    test_data = pd.read_csv('./test_only.csv')

    trainloader, num_stores = get_dataloader(train_data, opt.batch_size, G, index_neighbor_mapping_dict, graph_365, shuffle=True)

    testloader, num_stores = get_dataloader(test_data, opt.batch_size, G, index_neighbor_mapping_dict, graph_365, shuffle=False)

    num_types = 1

    return trainloader, testloader, num_types, num_stores
    # return trainloader, testloader



def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()
    #每次loop算的是一个batch的log likelihood, 一个sum batch就加起来了

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions

    def move_to_device(batch, device):
        # Function to move tensors to the specified device
        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return x  # If not a tensor, return as is

        # Apply to_device to each item in the batch
        return tuple(map(to_device, batch))

    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        # event_time, time_gap, event_type, event_demand, batch_placekey_list, batch_place_graph_seq = map(lambda x: x.to(opt.device), batch)

        # print(event_type,'event_type')
        event_time, time_gap, event_type, event_demand, event_demand_orignal, batch_placekey_list, batch_place_graph_seq = move_to_device(batch, opt.device)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time, event_demand, batch_placekey_list, batch_place_graph_seq)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type, event_demand)
        #在这个地方做的sum over stores in batch
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time)

        print('==========batch time_se  ===========', se.item())

        #demand prediction
        demand_se = Utils.demand_loss(prediction[2], event_demand_orignal)

        print('==========batch demand_se  ===========', demand_se.item())


        # SE is usually large, scale it to stabilize training
        scale_time_loss = 1
        scale_demand_loss = 1000
        loss = event_loss + pred_loss + se / scale_time_loss + demand_se
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]


    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


# def eval_epoch(model, validation_data, pred_loss_func, opt):
#     """ Epoch operation in evaluation phase. """

#     model.eval()

#     total_event_ll = 0  # cumulative event log-likelihood
#     total_time_se = 0  # cumulative time prediction squared-error
#     total_event_rate = 0  # cumulative number of correct prediction
#     total_num_event = 0  # number of total events
#     total_num_pred = 0  # number of predictions
#     with torch.no_grad():
#         for batch in tqdm(validation_data, mininterval=2,
#                           desc='  - (Validation) ', leave=False):
#             """ prepare data """
#             # event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
#             event_time, time_gap, event_type, event_demand, batch_placekey_list, batch_place_graph_seq = move_to_device(batch, opt.device)


#             """ forward """
#             enc_out, prediction = model(event_type, event_time, event_demand, batch_placekey_list, batch_place_graph_seq)

#             """ compute loss """
#             event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type, event_demand)
#             event_loss = -torch.sum(event_ll - non_event_ll)
#             _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
#             # time prediction
#             se = Utils.time_loss(prediction[1], event_time)

#             print('==========batch time_se  ===========', se.item())

#             #demand prediction
#             demand_se = Utils.demand_loss(prediction[2], event_demand)

#             print('==========batch demand_se  ===========', demand_se.item())

#                 """ note keeping """
#             total_event_ll += -event_loss.item()
#             total_time_se += se.item()
#             total_event_rate += pred_num.item()
#             total_num_event += event_type.ne(Constants.PAD).sum().item()
#             total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

#     rmse = np.sqrt(total_time_se / total_num_pred)
#     return total_event_ll / total_num_event, total_event_rate / total_num_predf, rmse


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        # start = time.time()
        # valid_event, valid_type, valid_time = eval_epoch(model, validation_data, pred_loss_func, opt)
        # print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
        #       'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
        #       'elapse: {elapse:3.3f} min'
        #       .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        # valid_event_losses += [valid_event]
        # valid_pred_losses += [valid_type]
        # valid_rmse += [valid_time]
        # print('  - [Info] Maximum ll: {event: 8.5f}, '
        #       'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
        #       .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        # # logging
        # with open(opt.log, 'a') as f:
        #     f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
        #             .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))

        scheduler.step()



def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-log', type=str, default='log.txt')

    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cpu')

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    POI_2019_non_sparse = pd.read_csv('./POI_2019_non_sparse.csv')

    dist_matrix = np.load('./dist_matrix.npy')

    placekey_index_mapping_dict = POI_2019_non_sparse['PLACEKEY'].to_dict()

    # whole_node_features_pool_non_sparse2 = pd.read_csv('./whole_node_features_pool_non_sparse2.csv')

    combined_df = pd.read_csv('./combined_df.csv')
    graph_365 = [create_dataframe_day(t, combined_df) for t in range(30, 365)]

    # print(graph_365[0],'graph_365 DataFrame 1')
    # print(graph_365[1],'graph_365 new_dataframe  2 ')

    # graph_365 = [create_dataframe_day(t, whole_node_features_pool_non_sparse2) for t in range(30, 365)]

    index_neighbor_mapping_dict = create_index_neighbor_mapping(dist_matrix)

    G, num_node_features = create_G(dist_matrix, POI_2019_non_sparse)

    collate_function_instance = CollateFunction(G, index_neighbor_mapping_dict, graph_365, placekey_index_mapping_dict)

    train_data = pd.read_csv('./test_only.csv')
    train_ds = EventData(train_data)
    
    trainloader = torch.utils.data.DataLoader(
        train_ds,
        num_workers=1,
        batch_size= opt.batch_size,
        collate_fn = collate_function_instance,
        shuffle= True
    )
    # num_stores = train_ds.length

    testloader = trainloader

    # trainloader, testloader, num_types, num_stores = prepare_dataloader(opt, G, index_neighbor_mapping_dict, graph_365)


    #如果没有store specific的b的话 不需要这个num_stores  这个num_stores最好是在model外面initailize, 如在predict那里单独intialize一个b class

    """ prepare model """
    #only one transformer is trained
    model = Transformer(
        num_types= 1,
        num_node_features = num_node_features,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )

    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, 1, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt)


if __name__ == '__main__':
    main()
