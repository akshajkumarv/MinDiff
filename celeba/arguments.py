import argparse

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('--device_id', type=str, default='0') 
parser.add_argument('--lr', type=float, default=0.0001) 
parser.add_argument('--steps', type =int, default=48000)
parser.add_argument('--run', type =int, default=1)
parser.add_argument('--flood_level', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=1.0)  
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--fnr_gap', type =float, default = 0.01)
parser.add_argument('--mindiff_weight', type=float, default=0.5) 
parser.add_argument('--mindiff', type=bool, default=True)

args = parser.parse_args()

