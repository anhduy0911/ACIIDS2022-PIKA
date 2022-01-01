import argparse
import config as CFG
from base_model import BaseModel
from kg_assisted_model import KGPillRecognitionModel
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(args):
    model = BaseModel(args)
    # model = KGPillRecognitionModel(args)
    model.train_v2()
    # model.evaluate_v2()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Torch KG-PR')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N', dest='batch_size')
    parser.add_argument('--val-batch-size', type=int, default=64, metavar='N', dest='v_batch_size')
    parser.add_argument('--train-folder', type=str,
                        default="data/pills/train/",
                        help='training folder path')
    parser.add_argument('--val-folder', type=str,
                        default="data/pills/test/",
                        help='validation folder path')
    parser.add_argument('--log-dir', type=str,
                        default="logs/runs/",
                        help='TensorBoard folder path')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', dest='epochs',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', dest='lr',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=911, metavar='S',
                        help='random seed (default: 911)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                        help='how many epoches to wait before saving model')
    parser.add_argument('--save-folder', type=str, default="logs/checkpoints", metavar='N',
                        help='how many epoches to wait before saving model')
    
    args = parser.parse_args()
    seed_everything(CFG.seed_number)
    main(args)