PATH = {
    'param': 'param',
    'data': 'data',
    'grid': 'data/population_and_employment/grid_1km_653_573/grid_cells.shp',
    'isochrones': 'data/isochrones/isochrones_json',
    'cache': 'data/sensor_cache',
    'population_and_employment': 'data/population_and_employment/population_and_employment.h5',
    'landuse_and_poi': 'data/landuse_poi/landuse_and_poi-230101.h5',
    'ground_truth': 'data/traffic_volume/traffic_volume_summary_2022.h5',
    'evaluate': 'eval/logs',
}

DATA = {
    'population': [0],
    'employment': [18],
    'landuse_poi': list(range(12)),
    'isochrone_intervals': list(range(300, 1201, 300)),
    'grid_resolution': 1000,
    'clear_cache': False,
}

MODEL = {
    'embedding_dim': 128,
    'patch_size': 5000,
    'num_attention_layers': 1,
    # 'num_attention_heads': 3,
    'pooling_method': 'attention',
    }

TRAINING = {
    'use_gpu': True,
    'seed': 4,
    'lr': 1e-4,
    'epoch': 5,
    'clip_gradient': 5,
    'train_prop': 0.8,
    # 'batch_size': 16,
    'loss_function': 'MGEH',
    'eval_metrics': ['MGEH'],
    'eval_samples': 50,
    'eval_interval': 1000,
    }
