import argparse

hp_parser = argparse.ArgumentParser(description='training')
# settings
hp_parser.add_argument('--gpu_id', type=int, help='gpu id to use')
hp_parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')

# hyperparameters
hp_parser.add_argument('--bce', type=float, default=1.0, help='weight for segmentation loss')
hp_parser.add_argument('--reg_uv', type=float, default=1.0, help='weight for uv map regression loss')
hp_parser.add_argument('--uv_loss', choices=['l1', 'l2', 'smoothl1'], default='smoothl1',
                       help='loss function used for uv map regression')
hp_parser.add_argument('--lm', type=float, default=1.0, help='weight for landmark regression loss')
hp_parser.add_argument('--k', type=int, default=5, help='number of nearest neighbors for landmark regression')
hp_parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
hp_parser.add_argument('--batch_size', type=int, default=10)
hp_parser.add_argument('--infer_batch_size', type=int, default=16, help='batch size during validation and testing')
hp_parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
hp_parser.add_argument('--lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='use learning rate scheduler')
hp_parser.add_argument('--beta', type=float, default=0.8, help='beta for smooth l1 loss')