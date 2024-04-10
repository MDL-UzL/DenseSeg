import argparse

hp_parser = argparse.ArgumentParser(description='training')
# settings
hp_parser.add_argument('--gpu_id', type=int, help='gpu id to use')
hp_parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
hp_parser.add_argument('--seg', action=argparse.BooleanOptionalAction, help='train segmentation head')
hp_parser.add_argument('--uv', action=argparse.BooleanOptionalAction, help='train uv head')

# hyperparameters
hp_parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
hp_parser.add_argument('--batch_size', type=int, default=8)
hp_parser.add_argument('--infer_batch_size', type=int, default=16, help='batch size during validation and testing')
hp_parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
hp_parser.add_argument('--lr_scheduler', action=argparse.BooleanOptionalAction, help='use learning rate scheduler')
hp_parser.add_argument('--uv_loss', choices=['l1', 'l2', 'smoothl1'], default='smoothl1', help='uv loss function')
hp_parser.add_argument('--beta', type=float, default=0.8, help='beta for smooth l1 loss')