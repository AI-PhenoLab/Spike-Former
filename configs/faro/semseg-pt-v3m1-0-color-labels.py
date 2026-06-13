_base_ = [
    "../_base_/default_runtime.py",
]

# misc custom setting - Optimized for CUDA 12.7 and low memory
batch_size = 32 # Increased to avoid BatchNorm error (needs at least 2 samples per batch)
num_worker = 0  # Windows compatibility: set to 0 to avoid multiprocessing issues
mix_prob = 0.8
empty_cache = True  # Enable to free GPU memory periodically
enable_amp = False  # Disable AMP for CUDA 12.7
eval_epoch = 50
enable_wandb = False  # Disable wandb to avoid login requirement

# model settings - Optimized for CUDA 12.7 and minimal memory
# 注意：num_classes 需要根据实际数据中的类别数调整
model = dict(
    type="DefaultSegmentorV2",
    num_classes=2,  # 根据实际数据中的类别数调整（例如：2, 3, 4...）
    backbone_out_channels=32,  # Match the last decoder channel (32)
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,  # XYZ + RGB
        order=("z", "z-trans"),  # Removed hilbert encodings to avoid depth limit
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 2, 2),  # Further reduced depth
        enc_channels=(16, 32, 64, 128, 256),  # Further reduced channels for lower memory
        enc_num_head=(1, 2, 4, 8, 16),  # Reduced attention heads
        enc_patch_size=(64, 64, 64, 64, 64),  # Further reduced patch size
        dec_depths=(2, 2, 2, 2),
        dec_channels=(32, 32, 64, 128),  # Further reduced channels
        dec_num_head=(2, 2, 4, 8),  # Reduced attention heads
        dec_patch_size=(64, 64, 64, 64),  # Further reduced patch size
        mlp_ratio=2,  # Keep at 2 to save memory
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=True,  # Enable RPE since Flash Attention is disabled
        enable_flash=False,  # Disabled for CUDA 12.7 compatibility
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
optimizer = dict(type="AdamW", lr=0.0005, weight_decay=0.01)
scheduler = dict(
    type="CosineAnnealingLR",
    # total_steps will be automatically calculated by build_scheduler
)

# dataset settings
dataset_type = "FARODataset"
# 数据路径：指向包含 data 子目录的根目录
# 例如：如果数据在 pointcept/datasets/data/ 下，则 data_root 应该指向 pointcept/datasets/
data_root = "pointcept/datasets/data"

# 类别数量（需要根据实际数据调整）
num_classes = 2  # 根据实际数据中的类别数调整

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=[f"class_{i}" for i in range(num_classes)],  # 通用类别名称，可以根据需要修改
    train=dict(
        type=dataset_type,
        split="train",  # 注意：使用颜色标签时，split 参数会被忽略，直接从 data 目录读取
        data_root=data_root,
        use_color_labels=True,  # 启用颜色标签模式
        num_classes=num_classes,
        color_mapping_mode="centers",  # 使用颜色中心模式（自动识别红蓝）

        color_centers=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # 例如：红色=类别0，蓝色=类别1
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
                grid_size=0.05,  # Further increased to prevent serialization depth > 16 error
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # Removed SphereCrop to process all points without point_max limit
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),  # Include segment for training
                feat_keys=("coord", "color"),  # XYZ + RGB = 6 channels
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",  # 注意：使用颜色标签时，split 参数会被忽略，直接从 data 目录读取
        data_root=data_root,
        use_color_labels=True,  # 启用颜色标签模式
        num_classes=num_classes,
        color_mapping_mode="centers",  # ✅ 显式设置
        color_centers=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        # color_mapping_mode="threshold",  # 使用与训练集相同的映射模式
        # color_threshold=0.5,
        # color_channel="mean",
        # threshold_low_class=0,
        # threshold_high_class=1,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.05,  # Further increased to prevent serialization depth > 16 error
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
                keys=("coord", "grid_coord", "segment"),  # Include segment for validation
                feat_keys=("coord", "color"),  # XYZ + RGB = 6 channels
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",  # 注意：使用颜色标签时，split 参数会被忽略，直接从 data 目录读取
        data_root=data_root,
        use_color_labels=True,  # 启用颜色标签模式
        num_classes=num_classes,
        color_mapping_mode="centers",  # ✅ 显式设置
        color_centers=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        # color_mapping_mode="threshold",  # 使用与训练集相同的映射模式
        # color_threshold=0.5,
        # color_channel="mean",
        # threshold_low_class=0,
        # threshold_high_class=1,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,  # Further increased to prevent serialization depth > 16 error
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

