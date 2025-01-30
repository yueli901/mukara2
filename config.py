PATH = {
    'param': 'param',
    'data': 'data',
    'grid': 'data/population_and_employment/grid_1km_653_573/grid_cells.shp',
    'isochrones': 'data/isochrones/isochrones_json',
    'roi': 'data/roi_cache',
    'population_and_employment': 'data/population_and_employment/population_and_employment.h5',
    'landuse_and_poi': 'data/landuse_poi/landuse_and_poi-230101.h5',
    'ground_truth': 'data/traffic_volume/traffic_volume_summary_2022.h5',
    'evaluate': 'eval/logs',
}

DATA = {
    'population': [0],
    'employment': [18],
    'landuse_poi': list(range(12)),
    'isochrone_intervals': [300, 600, 900, 1800, 3600],
    'grid_resolution': 1000,
    'clear_cache': False,
}

MODEL = {
    # Model name
    'model': 'cnn',
    # MLP
    'activation': 'relu',
    'hiddens': [128],
    'output': [128],
    # CNN
    'depth_cnn': 5,
    'channels': [16, 32, 64, 128, 256],
    'kernel_size': 3,
    'strides': 1, 
    'pool_size': 2,
    'pool_strides': 2,
    'output_dense': 128,
    }

TRAINING = {
    'seed': 4,
    'lr': 1e-3,
    'epoch': 10,
    'step': 32,
    'clip_gradient': 5,
    'train_prop': 0.9,
    'batch_size': 32,
    'loss_function': 'MGEH',
    'eval_metrics': ['MGEH', 'MAE'],
    'eval_samples': 50,
    }
