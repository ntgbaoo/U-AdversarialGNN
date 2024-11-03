import torch


if __name__ == "__main__":
    edge_list = torch.tensor([
        [1,2,3,7],
        [4,5,6,8]
    ])
    print(edge_list.shape)

    potential_nodes = [9,10]
    target_node = 2
    for node in potential_nodes:
        new_edge = torch.tensor([node,target_node])
        new_edge = new_edge.unsqueeze(dim=1)
        edge_list = torch.cat((edge_list,new_edge),dim= 1)
    
    print(edge_list)
    print(edge_list.shape)