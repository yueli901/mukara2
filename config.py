PATH = {
    'param': 'param',
    'data': 'data',
    'mesh': 'data/population_and_employment/grid_1km_653_573/grid_cells.shp',
    'graph_nodes': 'data/highway_network/england_wales_nodes_27700.json',
    'graph_edges': 'data/highway_network/england_wales_edges_27700.json',
    'nodes': 'data/highway_network/nodes.json',
    'edge_features': 'data/highway_network/edge_features.json',
    'mesh2nodes': 'data/highway_network/mesh_nodes_pairing.json',
    'mesh2edges': 'data/highway_network/mesh_edges_pairing.json',
    'cache': 'data/cache',
    'population_and_employment': 'data/population_and_employment/population_and_employment.h5',
    'landuse_and_poi': 'data/landuse_poi/landuse_and_poi-230101.h5',
    'ground_truth': 'data/traffic_volume/traffic_volume_by_edges.csv',
    'evaluate': 'eval/logs',
}

DATA = {
    'population': [0],
    'employment': [18],
    'landuse_poi': list(range(12)),
    'clear_cache': False,
}

MODEL = {
    'embedding_dim': 128,
    'gnn_layers': 64, # degrees of neighbours
    'attention_heads': 3,
    }

TRAINING = {
    'use_gpu': True,
    'seed': 1,
    'lr': 1e-4,
    'epoch': 5,
    'clip_gradient': 5,
    'train_prop': 0.8,
    'loss_function': 'MSE_Z',
    'eval_metrics': ['MGEH', 'MAE'],
    'eval_interval': 1,
    }
