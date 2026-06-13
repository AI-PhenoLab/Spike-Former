_base_ = [
    "../_base_/default_runtime.py",
]

# misc custom setting
batch_size = 1  # bs: total bs in all gpus (reduced for Windows/CUDA 12.7)
num_worker = 0  # Windows compatibility: set to 0 to avoid multiprocessing issues
mix_prob = 0.8
empty_cache = False
enable_amp = False  # Disable AMP for CUDA 12.7 compatibility
eval_epoch = 1  # Evaluate every epoch
enable_wandb = False  # Disable wandb to avoid login requirement

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=1,  # Change this if you have semantic segmentation labels
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,  # XYZ + RGB
        order=("z", "z-trans"),  # Removed hilbert encodings to avoid depth limit
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(512, 512, 512, 512, 512),  # Reduced for CUDA 12.7 compatibility
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(512, 512, 512, 512),  # Reduced for CUDA 12.7 compatibility
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=True,  # Enable RPE since Flash Attention is disabled
        enable_flash=False,  # Disable Flash Attention for CUDA 12.7 compatibility
        upcast_attention=True,  # Use fp32 precision
        upcast_softmax=True,  # Use fp32 precision
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
epoch = 200
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler = dict(
    type="CosineAnnealingLR",
    # total_steps will be automatically calculated by build_scheduler
)

# dataset settings
dataset_type = "FARODataset"
data_root = "pointcept/datasets/0329_FARO_part_1"

data = dict(
    num_classes=1,  # Change this if you have semantic segmentation labels
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.05,  # Increased from 0.02 to reduce grid_coord range
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=204800, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord"),  # Remove "segment" if you don't have labels
                feat_keys=("coord", "color"),  # XYZ + RGB = 6 channels
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.05,  # Increased from 0.02 to reduce grid_coord range
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord"),  # Remove "segment" if you don't have labels
                feat_keys=("coord", "color"),  # XYZ + RGB = 6 channels
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,  # Increased from 0.02 to reduce grid_coord range
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color"),  # XYZ + RGB = 6 channels
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
            ],
        ),
    ),
)