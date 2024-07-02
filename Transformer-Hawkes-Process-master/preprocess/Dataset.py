import numpy as np
import torch
import torch.utils.data

from transformer import Constants
import ast


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
      
        self.placekey_list = list(data['placekey'])
        print(self.placekey_list, 'placekey')

        all_store_events = list(data['event_sequence'])

        all_store_events = [ast.literal_eval(event_string) for event_string in data['event_sequence']]

        self.time = [[single_event['time'] for single_event in single_store_events] for single_store_events in all_store_events]
        self.time_gap = [[single_event['time_since_last_event'] for single_event in single_store_events] for single_store_events in all_store_events]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.demand_marker = [[single_event['actual_demand'] + 0.00001 for single_event in single_store_events] for single_store_events in all_store_events]
        # plus 1 since there could be event type 0, but we use 0 as padding
        self.event_type = [[1 for single_event in single_store_events] for single_store_events in all_store_events]

        self.length = len(all_store_events)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx], self.demand_marker[idx], self.placekey_list[idx]


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


def create_dataframe_day(t, dataframe):
    start_idx = t - 30
    end_idx = t

    new_dataframe = pd.DataFrame()
    new_dataframe['spend'] = dataframe['spend'].apply(lambda x: x[start_idx:end_idx])

    new_dataframe['placekey'] = dataframe['PLACKEY']

    return new_dataframe

def add_demand_features(sub_graph, spend_df, placekey_index_list):
    # Assume `data` is a PyTorch Geometric Data object
    for index in placekey_index_list:
      placekey = index_placekey_mapping_dict[index]
      spend_data = spend_df.loc[spend_df['PLACEKEY'] == placekey, 'spend'].values
      spend_tensor = torch.tensor(spend_data, dtype=torch.float)
      # Ensure the feature tensor is of proper shape, reshape if necessary
      spend_tensor = spend_tensor.view(1, -1)  # Reshape to 1 row, many columns
      existing_features = sub_graph.x[index]

      new_features = torch.cat([existing_features, spend_tensor], dim=0)
      sub_graph.x[index] = new_features

    return sub_graph

def create_ego_centric_graph(data, placekey_index):
    # Get the neighbors + the node itself
    neighbors = torch.cat([data.edge_index[0][data.edge_index[1] == placekey_index],
                           data.edge_index[1][data.edge_index[0] == placekey_index],
                           torch.tensor([placekey_index], device=data.edge_index.device)])
    neighbors = torch.unique(neighbors)  # Remove duplicates

    # Extract the subgraph
    sub_data = subgraph(neighbors, data.edge_index, num_nodes=data.num_nodes, edge_attr=data.edge_attr, relabel_nodes = False)
    sub_data.x = data.x[neighbors]

    return sub_data


# def collate_fn(all_store_events, G, index_neighbor_mapping_dict, graph_365):
def collate_fn(all_store_events):

    #all_store_events就是batch

    #all store_events这里应该是 (time, type, demand, plackey) x batch size这样的一个一个的了
    #第一行的目的是恢复那几个list

    #batch_placekey_list的顺序应与poi non sparse一致， 这样就和dist_matrix一致了
    # time, time_gap, event_type, event_demand, batch_placekey_list = list(zip(*all_store_events))
    time, time_gap, event_type, event_demand, batch_placekey_list = [list(x) for x in zip(*all_store_events)]

    all_place_graph_seq = []
    print(batch_placekey_list)
    print(time)
    for i in range(len(batch_placekey_list)):
      egocentric_graph = create_ego_centric_graph(G, i)
      #time sequence for current placekey
      time_seq = time[i]
      graph_sequence_current_placekey =[]


      #要写这个placekey_index_mapping dict
      placekey = placeykey_index_mapping_dict[i]
      for t in time_seq:
        #whole graph dataframe at time t
        graph_dataframe_t = graph_365[t]
        placekey_index_list = index_neighbor_mapping_dict[i]
        new_graph = add_demand_features(egocentric_graph, graph_dataframe_t, placekey_index_list)
        graph_sequence_current_placekey.append(new_graph)
      all_place_graph_seq.append(graph_sequence_current_placekey)

    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_demand = pad_demand_mark(event_demand)
    event_type = pad_type(event_type)
    batch_placekey_list = pad_placekey(batch_placekey_list)
    all_place_graph_seq = pad_graph(all_place_graph_seq)
    return time, time_gap, event_type, event_demand, batch_placekey_list, all_place_graph_seq


def get_dataloader(data, batch_size, G, index_neighbor_mapping_dict, graph_365, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        # collate_fn= lambda x: collate_fn(x, G, index_neighbor_mapping_dict, graph_365),
        collate_fn = collate_fn,
        shuffle=shuffle
    )
    num_stores = ds.length
    return dl, num_stores


