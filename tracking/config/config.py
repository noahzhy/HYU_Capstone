# config.py
cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'anchorNum_per_stage': 2,
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': False,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': [640, 640],
    'coco': False,
    'n_class': 12,
    'pretrain': False,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 128
}

cfg_shuffle = {
    'name': 'ShuffleNetG2',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'anchorNum_per_stage': 2,
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': False,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': [640, 640],
    'pretrain': False,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'ShuffleNetG2_return_layers': {'layer1': 1, 'layer2': 2, 'layer3': 3},
    'in_channel': 100,
    'out_channel': 256,
    'ShuffleNetG2': {
        'out_planes': [200, 400, 800],
        'num_blocks': [4, 8, 4],
        'groups': 2
    }
}

cfg_shufflev2 = {
    'name': 'ShuffleNetV2',
    'min_sizes': [[24, 8], [48, 16], [96, 32], [384, 128]],
    # 'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'anchorNum_per_stage': 2,
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': False,
    'batch_size': 8,
    'ngpu': 4,
    'epoch': 100,
    'image_size': [640, 640],
    'pretrain': False,
    'ShuffleNetV2_return_layers': {'layer1': 1, 'layer2': 2, 'layer3': 3},
    'in_channel': 58,
    'out_channel': 128,
    'n_class': 12,
    'coco': False,
    'num_classes': 80,
    'ShuffleNetV2': {
        'out_planes': [200, 400, 800],
        'stage_repeats': [4, 8, 4],
        'groups': 2,
        'image_size': [640, 640],
        'width_mult': 1.5,  # 缩放系数
        'n_class': 12
    }
}
