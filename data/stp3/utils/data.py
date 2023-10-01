import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import io
import copy
from io import BytesIO
import matplotlib
import matplotlib as mpl
import PIL
from PIL import Image

from stp3.utils.geometry import cumulative_warp_features_reverse, cumulative_warp_features, extract_trajs, extract_obs_from_centerness, generate_instance_colours

def prepare_future_labels(batch, model):
    labels = {}

    cfg = model.cfg

    segmentation_labels = batch['segmentation']
    hdmap_labels = batch['hdmap']
    future_egomotion = batch['future_egomotion']
    gt_trajectory = batch['gt_trajectory']
    labels['sample_trajectory'] = batch['sample_trajectory']

    # present frame hd map gt
    labels['hdmap'] = hdmap_labels[:, model.receptive_field - 1].long().contiguous()

    # first rec token
    if cfg.DATASET.VERSION == 'nuscenes':
        labels['rec_first'] = batch['rec_first']
        labels['gt_trajectory_prev'] = batch['gt_trajectory_prev']

    # gt trajectory
    labels['gt_trajectory'] = gt_trajectory
    spatial_extent = (cfg.LIFT.X_BOUND[1], cfg.LIFT.Y_BOUND[1])

    # Past frames gt depth
    if cfg.LIFT.GT_DEPTH:
        depths = batch['depths']
        depth_labels = depths[:, :model.receptive_field, :, ::model.encoder_downsample,
                        ::model.encoder_downsample]
        depth_labels = torch.clamp(depth_labels, cfg.LIFT.D_BOUND[0], cfg.LIFT.D_BOUND[1] - 1) - \
                        cfg.LIFT.D_BOUND[0]
        depth_labels = depth_labels.long().contiguous()
        labels['depths'] = depth_labels

    # Warp labels to present's reference frame
    segmentation_labels_past = cumulative_warp_features(
        segmentation_labels[:, :model.receptive_field].float(),
        future_egomotion[:, :model.receptive_field],
        mode='nearest', spatial_extent=spatial_extent,
    ).long().contiguous()[:, :-1]
    segmentation_labels = cumulative_warp_features_reverse(
        segmentation_labels[:, (model.receptive_field - 1):].float(),
        future_egomotion[:, (model.receptive_field - 1):],
        mode='nearest', spatial_extent=spatial_extent,
    ).long().contiguous()
    labels['segmentation'] = torch.cat([segmentation_labels_past, segmentation_labels], dim=1)

    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        hdmap_labels = batch['hdmap'][:, :, 1:2]
        hdmap_labels_past = cumulative_warp_features(
            hdmap_labels[:, :model.receptive_field].float(),
            future_egomotion[:, :model.receptive_field],
            mode='nearest', spatial_extent=spatial_extent,
        ).long().contiguous()[:, :-1]
        hdmap_labels = cumulative_warp_features_reverse(
            hdmap_labels[:, (model.receptive_field - 1):].float(),
            future_egomotion[:, (model.receptive_field - 1):],
            mode='nearest', spatial_extent=spatial_extent,
        ).long().contiguous()
        labels['hdmap_warped_road'] = torch.cat([hdmap_labels_past, hdmap_labels], dim=1)
        hdmap_labels = batch['hdmap'][:, :, 0:1]
        hdmap_labels_past = cumulative_warp_features(
            hdmap_labels[:, :model.receptive_field].float(),
            future_egomotion[:, :model.receptive_field],
            mode='nearest', spatial_extent=spatial_extent,
        ).long().contiguous()[:, :-1]
        hdmap_labels = cumulative_warp_features_reverse(
            hdmap_labels[:, (model.receptive_field - 1):].float(),
            future_egomotion[:, (model.receptive_field - 1):],
            mode='nearest', spatial_extent=spatial_extent,
        ).long().contiguous()
        labels['hdmap_warped_lane'] = torch.cat([hdmap_labels_past, hdmap_labels], dim=1)            

    return labels
