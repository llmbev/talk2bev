import argparse
from fvcore.common.config import CfgNode as _CfgNode


def convert_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary."""
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, _CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                'Key {} with value {} is not a valid type; valid types: {}'.format(
                    '.'.join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


class CfgNode(_CfgNode):
    """Remove once https://github.com/rbgirshick/yacs/issues/19 is merged."""

    def convert_to_dict(self):
        return convert_to_dict(self)


CN = CfgNode

_C = CN()
_C.LOG_DIR = 'tensorboard_logs'
_C.TAG = 'default'

_C.GPUS = [0, 1, 2]  # which gpus to use
_C.PRECISION = 32  # 16bit or 32bit
_C.BATCHSIZE = 1
_C.EPOCHS = 20
_C.DEBUG = True
_C.DENOISE = False # to denoise the GT or not
_C.LOGGER = 'wandb'
_C.COMPARE_WITH_ORIG = True # to compare with original or not

_C.N_WORKERS = 0
_C.VIS_INTERVAL = 5000
_C.LOGGING_INTERVAL = 500

_C.PRETRAINED = CN()
_C.PRETRAINED.LOAD_WEIGHTS = True
_C.PRETRAINED.PATH = 'last.ckpt'

_C.DATASET = CN()
_C.DATASET.DATAROOT = '/mnt/e/datasets/nuscenes'
_C.DATASET.VERSION = 'trainval'
_C.DATASET.NAME = 'nuscenes'
_C.DATASET.MAP_FOLDER = "/scratch/tushar.choudhary/forecasting/v1.0-trainval"
_C.DATASET.IGNORE_INDEX = 255  # Ignore index when creating flow/offset labels
_C.DATASET.FILTER_INVISIBLE_VEHICLES = True  # Filter vehicles that are not visible from the cameras
_C.DATASET.SAVE_DIR = 'datas'
_C.DATASET.THRESHOLD = 25

_C.TIME_RECEPTIVE_FIELD = 3  # how many frames of temporal context (1 for single timeframe)
_C.N_FUTURE_FRAMES = 6  # how many time steps into the future to predict

_C.IMAGE = CN()
_C.IMAGE.FINAL_DIM = (224, 480)
_C.IMAGE.RESIZE_SCALE = 0.3
_C.IMAGE.TOP_CROP = 46
_C.IMAGE.ORIGINAL_HEIGHT = 900  # Original input RGB camera height
_C.IMAGE.ORIGINAL_WIDTH = 1600  # Original input RGB camera width
_C.IMAGE.NAMES = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

_C.LIFT = CN()  # image to BEV lifting
_C.LIFT.X_BOUND = [-50.0, 50.0, 0.5]  # Forward
_C.LIFT.Y_BOUND = [-50.0, 50.0, 0.5]  # Sides
_C.LIFT.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
_C.LIFT.D_BOUND = [2.0, 50.0, 1.0]
_C.LIFT.GT_DEPTH = False
_C.LIFT.DISCOUNT = 0.5

_C.EGO = CN()
_C.EGO.WIDTH = 1.85
_C.EGO.HEIGHT = 4.084

_C.MODEL = CN()

_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.DOWNSAMPLE = 8
_C.MODEL.ENCODER.NAME = 'efficientnet-b4'
_C.MODEL.ENCODER.OUT_CHANNELS = 64
_C.MODEL.ENCODER.USE_DEPTH_DISTRIBUTION = True

_C.MODEL.TEMPORAL_MODEL = CN()
_C.MODEL.TEMPORAL_MODEL.NAME = 'temporal_block'  # type of temporal model
_C.MODEL.TEMPORAL_MODEL.START_OUT_CHANNELS = 64
_C.MODEL.TEMPORAL_MODEL.EXTRA_IN_CHANNELS = 0
_C.MODEL.TEMPORAL_MODEL.INBETWEEN_LAYERS = 0
_C.MODEL.TEMPORAL_MODEL.PYRAMID_POOLING = True
_C.MODEL.TEMPORAL_MODEL.INPUT_EGOPOSE = True

_C.MODEL.DISTRIBUTION = CN()
_C.MODEL.DISTRIBUTION.LATENT_DIM = 32
_C.MODEL.DISTRIBUTION.MIN_LOG_SIGMA = -5.0
_C.MODEL.DISTRIBUTION.MAX_LOG_SIGMA = 5.0

_C.MODEL.FUTURE_PRED = CN()
_C.MODEL.FUTURE_PRED.N_GRU_BLOCKS = 2
_C.MODEL.FUTURE_PRED.N_RES_LAYERS = 1
_C.MODEL.FUTURE_PRED.MIXTURE = True

_C.MODEL.DECODER = CN()

_C.MODEL.BN_MOMENTUM = 0.1

_C.SEMANTIC_SEG = CN()

_C.SEMANTIC_SEG.VEHICLE = CN()
_C.SEMANTIC_SEG.VEHICLE.WEIGHTS = [1.0, 2.0]
_C.SEMANTIC_SEG.VEHICLE.USE_TOP_K = True  # backprop only top-k hardest pixels
_C.SEMANTIC_SEG.VEHICLE.TOP_K_RATIO = 0.25

_C.SEMANTIC_SEG.PEDESTRIAN = CN()
_C.SEMANTIC_SEG.PEDESTRIAN.ENABLED = True
_C.SEMANTIC_SEG.PEDESTRIAN.WEIGHTS = [1.0, 10.0]
_C.SEMANTIC_SEG.PEDESTRIAN.USE_TOP_K = True
_C.SEMANTIC_SEG.PEDESTRIAN.TOP_K_RATIO = 0.25

_C.SEMANTIC_SEG.HDMAP = CN()
_C.SEMANTIC_SEG.HDMAP.ENABLED = True
_C.SEMANTIC_SEG.HDMAP.ELEMENTS = ['lane_divider', 'drivable_area']
_C.SEMANTIC_SEG.HDMAP.WEIGHTS = [[1.0, 5.0], [1.0, 1.0]]
_C.SEMANTIC_SEG.HDMAP.TRAIN_WEIGHT = [1, 1]
_C.SEMANTIC_SEG.HDMAP.USE_TOP_K = [True, False]
_C.SEMANTIC_SEG.HDMAP.TOP_K_RATIO = [0.25, 0.25]

_C.INSTANCE_SEG = CN()
_C.INSTANCE_SEG.ENABLED = False

_C.INSTANCE_FLOW = CN()
_C.INSTANCE_FLOW.ENABLED = False

_C.PROBABILISTIC = CN()
_C.PROBABILISTIC.ENABLED = True  # learn a distribution over futures
_C.PROBABILISTIC.METHOD = 'GAUSSIAN' # [BERNOULLI, GAUSSIAN, MIXGAUSSIAN]

_C.PLANNING = CN()
_C.PLANNING.ENABLED = True
_C.PLANNING.GRU_ENABLED = False
_C.PLANNING.GRU_STATE_SIZE = 256
_C.PLANNING.SAMPLE_NUM = 150
_C.PLANNING.COMMAND = ['LEFT', 'FORWARD', 'RIGHT']

# dense configs
_C.PLANNING.DENSE = CN()
_C.PLANNING.DENSE.ENABLED = True # to consider dense centerlines
_C.PLANNING.DENSE.OBS = False # avoid obstacles
_C.PLANNING.DENSE.NUM_OBS = 5 # avoid obstacles
_C.PLANNING.DENSE.PATH = '../' # centerlines path

_C.FUTURE_DISCOUNT = 0.95

_C.OPTIMIZER = CN()
_C.OPTIMIZER.LR = 1e-3
_C.OPTIMIZER.WEIGHT_DECAY = 1e-7
_C.GRAD_NORM_CLIP = 5

_C.COST_FUNCTION = CN()
_C.COST_FUNCTION.SAFETY = 0.1
_C.COST_FUNCTION.LAMBDA = 1.
_C.COST_FUNCTION.HEADWAY = 1.
_C.COST_FUNCTION.LRDIVIDER = 10. 
_C.COST_FUNCTION.COMFORT = 0.1
_C.COST_FUNCTION.PROGRESS = 0.5
_C.COST_FUNCTION.VOLUME = 100.

def get_parser():
    parser = argparse.ArgumentParser(description='Fiery training')
    parser.add_argument('--config-file', default='', metavar='FILE', help='path to config file')
    parser.add_argument(
        'opts', help='Modify config options using the command-line', default=None, nargs=argparse.REMAINDER,
    )
    return parser


def get_cfg(args=None, cfg_dict=None):
    """ First get default config. Then merge cfg_dict. Then merge according to args. """

    cfg = _C.clone()

    if cfg_dict is not None:
        tmp = CfgNode(cfg_dict)
        for i in tmp.COST_FUNCTION:
            tmp.COST_FUNCTION.update({i: float(tmp.COST_FUNCTION.get(i))})
        cfg.merge_from_other_cfg(tmp)

    if args is not None:
        if args.config_file:
            cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # cfg.freeze()
    return cfg
