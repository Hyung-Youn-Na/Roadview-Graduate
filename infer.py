import argparse
import os

import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.custom_lm as module_arch
from parse_config import ConfigParser


def infer(config):
    logger = config.get_logger('test')

    str = input()

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)


    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    while str != 'END':
        output = model((str,))

        pred = torch.argmax(output, dim=1)
        print(str, len(str), output, pred)
        str = input()



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    infer(config)
