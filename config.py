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
    'isochrone_intervals': list(range(300, 3601, 300)),
    'grid_resolution': 1000,
    'clear_cache': False,
}

MODEL = {
    'embedding_dim': 128,
    'num_clusters': 30,
    'num_attention_layers': 2,
    # 'num_attention_heads': 3,
    'pooling_method': 'attention',
    }

TRAINING = {
    'lambda_traffic': 1,
    'lambda_compact': 0,
    'lambda_separate': 0,
    'lambda_balance': 0,

    'seed': 1,
    'lr': 1e-3,
    'epoch': 20,
    'step': 32,
    'clip_gradient': 5,
    'train_prop': 0.8,
    'batch_size': 16,
    'loss_function': 'MGEH',
    'eval_metrics': ['MGEH'],
    'eval_samples': 50,
    'eval_interval': 1000,
    }
