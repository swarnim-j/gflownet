"""
Runnable script with hydra capabilities
"""
import os
import pickle
import random
import sys

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


@hydra.main(config_path="./config", config_name="main", version_base="1.1")
def main(config):
    # Get current directory and set it as root log dir for Logger
    cwd = os.getcwd()
    config.logger.logdir.root = cwd
    print(f"\nLogging directory of this run:  {cwd}\n")
    sys.path.append('../mcrygan')

    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)

    # Logger
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)
    # The proxy is required in the env for scoring: might be an oracle or a model
    proxy = hydra.utils.instantiate(
        config.proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    # The proxy is passed to env and used for computing rewards
    env = hydra.utils.instantiate(
        config.env,
        proxy=proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    if 'MolCrystal' in str(env.__class__):
        dataDims = env.proxy.model.dataDims
        config.gflownet['policy']['forward']['n_node_feats'] = dataDims['num atom features']
        config.gflownet['policy']['forward']['n_graph_feats'] = dataDims['num mol features'] - dataDims['num crystal generation features']
        config.gflownet['policy']['forward']['max_mol_radius'] = 5  # todo de-hard-code
        config.gflownet['policy']['forward']['n_crystal_features'] = dataDims['num crystal generation features']
        config.gflownet['policy']['backward']['n_node_feats'] = dataDims['num atom features']
        config.gflownet['policy']['backward']['n_graph_feats'] = dataDims['num mol features'] - dataDims['num crystal generation features']
        config.gflownet['policy']['backward']['max_mol_radius'] = 5  # todo de-hard-code
        config.gflownet['policy']['backward']['n_crystal_features'] = dataDims['num crystal generation features']

    gflownet = hydra.utils.instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env=env,
        buffer=config.env.buffer,
        logger=logger,
        machine = config.machine,
    )
    gflownet.train()

    # Sample from trained GFlowNet
    if config.n_samples > 0 and config.n_samples <= 1e5:
        batch, times = gflownet.sample_batch(n_forward=config.n_samples, train=False)
        x_sampled = batch.get_terminating_states(proxy=True)
        energies = env.oracle(x_sampled)
        x_sampled = batch.get_terminating_states()
        df = pd.DataFrame(
            {
                "readable": [env.state2readable(x) for x in x_sampled],
                "energies": energies.tolist(),
            }
        )
        df.to_csv("gfn_samples.csv")
        dct = {"x": x_sampled, "energy": energies}
        pickle.dump(dct, open("gfn_samples.pkl", "wb"))

    # Print replay buffer
    print(gflownet.buffer.replay)

    # Close logger
    gflownet.logger.end()


def set_seeds(seed):
    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
    sys.exit()
