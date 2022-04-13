import argparse
import config as CFG
from base_model import BaseModel
from base_model_hancraft import BaseModel as BaseModelHandCraft
from kg_e2e import KGPillRecognitionModel
from kg_assisted_model import KGPillRecognitionModel as KGPillRecognitionModel_assisted
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
    # model = BaseModel(args)
    # model = BaseModelHandCraft(args)
    # model = KGPillRecognitionModel(args)
    model = KGPillRecognitionModel_assisted(args)
    if args.mode == 'train':
        model.train()
        # model.evaluate()
    else:
        model.evaluate()
    # model.save_cpu(best=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Torch KG-PR')

    parser.add_argument('--batch-size', type=int, default=32, metavar='N', dest='batch_size')
    parser.add_argument('--val-batch-size', type=int, default=32, metavar='N', dest='v_batch_size')
    parser.add_argument('--train-folder', type=str,
                        default="data/pills/train_new/",
                        help='training folder path')
    parser.add_argument('--val-folder', type=str,
                        default="data/pills/test_new/",
                        help='validation folder path')
    parser.add_argument('--log-dir', type=str,
                        default="logs/runs/",
                        help='Log folder path')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', dest='epochs',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', dest='lr',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=911, metavar='S',
                        help='random seed (default: 911)')
    parser.add_argument('--name', type=str, default="baseline", metavar='N',
                        help='name of saving model')
    parser.add_argument('--patience', type=int, default=20, metavar='N',
                        help='patience of early stopping')
    parser.add_argument('--backbone', type=str, default="resnet50", metavar='N', help='choose backbone model')
    parser.add_argument('--loss', type=str, default="js", metavar='N', help='choose loss for projection module')
    parser.add_argument('--g_emd_path', type=str, default="./data/converted_graph/condened_g_embedding_deepwalk_w.json", metavar='N')
    parser.add_argument('--exclude_path', type=str, default="", metavar='N')
    parser.add_argument('--warmstart_backbone', type=bool, default=False, metavar='N')
    parser.add_argument('--mode', type=str, default="train", metavar='N')
    
    
    args = parser.parse_args()
    print(args)
    seed_everything(CFG.seed_number)
    main(args)