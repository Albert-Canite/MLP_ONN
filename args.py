import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--image_size',choices=[7,14,28] ,type=int,default=14)
parser.add_argument('--layer_num',type=int,choices=[1,2,3,4],default=4)
parser.add_argument('--train_epoch',type=int,default=20)
parser.add_argument('--hidden_layer_size',type=int,default=14)
parser.add_argument('--distill_epoch',type=int,default=5)
parser.add_argument('--prune_amount',type=float,default=0.5)
parser.add_argument('--input_noise_std', type=float, default=0.1,
                    help='Gaussian noise std applied to training inputs after ToTensor (10% by default).')
parser.add_argument('--gradient_noise_std', type=float, default=0.1,
                    help='Gaussian noise ratio applied during backward gradient propagation (10% by default).')
parser.add_argument('--gradient_num_bits', type=int, default=5,
                    help='Bit width used to quantize gradients during backward propagation.')
args = parser.parse_args()
