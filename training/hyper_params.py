import argparse

hp_parser = argparse.ArgumentParser(description='training')
# settings
hp_parser.add_argument('--gpu_id', type=int, help='gpu id to use')
hp_parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')

# hyperparameters
hp_parser.add_argument('--bce', type=float, default=1, help='weight for segmentation loss')
hp_parser.add_argument('--reg_uv', type=float, default=1, help='weight for uv map regression loss')
hp_parser.add_argument('--uv_loss', choices=['l1', 'l2', 'smoothl1'], default='l1',
                       help='loss function used for uv map regression')
hp_parser.add_argument('--tv', type=float, default=1, help='weight for total variation loss')
hp_parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
hp_parser.add_argument('--batch_size', type=int, default=8)
hp_parser.add_argument('--infer_batch_size', type=int, default=8, help='batch size during validation and testing')
hp_parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
hp_parser.add_argument('--lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='use learning rate scheduler')
hp_parser.add_argument('--beta', type=float, default=0.8, help='beta for smooth l1 loss')

# random affine augmentation
hp_parser.add_argument('--rotate', type=float, default=25, help='rotation angle in degrees')
hp_parser.add_argument('--translate', type=float, default=0.2, help='translation factor')
hp_parser.add_argument('--scale', type=float, default=0.3, help='scaling factor')