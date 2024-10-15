import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color

from BUILD import BUILD
from data.dataset import BuildDataset


def finetune(dataset, pretrained_file, fix_enc, **kwargs):
    # configurations initialization
    props = ['props/BUILD.yaml', 'props/finetune.yaml']
    print(props)

    # configurations initialization
    config = Config(model=BUILD, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = BuildDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = BUILD(config, train_data.dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')

        model.load_state_dict(checkpoint['state_dict'], strict=False)

        # Filter out unnecessary keys by checking the prefix for MoEAdaptorLayer
        # moe_keys = {k: v for k, v in checkpoint['state_dict'].items() if 'moe_adaptor' in k}
        # # Load only MoEAdaptorLayer parameters
        # model_dict = model.state_dict()
        # model_dict.update(moe_keys)  # Update model's state dict only with MoEAdaptorLayer parameters
        # model.load_state_dict(model_dict, strict=False)  # strict=False to allow for incomplete state dict

        if fix_enc == 'fix_enc':
            logger.info(f'Fix encoder parameters.')
            for _ in model.position_embedding.parameters():
                _.requires_grad = False
            for _ in model.trm_encoder.parameters():
                _.requires_grad = False
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Scientific', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    parser.add_argument('-f', type=str, default='', help='fine-tune mode')
    args, unparsed = parser.parse_known_args()
    print(args)
    finetune(args.d, pretrained_file=args.p, fix_enc=args.f)
