from argparse import ArgumentParser

import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch

from utils.common import instantiate_from_config, load_state_dict
from pytorch_lightning.loggers import CSVLogger

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--refresh_rate", type=int, default=1)
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    pl.seed_everything(config.lightning.seed, workers=True)
        
    data_module = instantiate_from_config(config.data)
    
    model = instantiate_from_config(OmegaConf.load(config.model.config))
    # TODO: resume states saved in checkpoint.
    if config.model.get("resume"):
        load_state_dict(model, torch.load(config.model.resume, map_location="cpu"), strict=False)
    
    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))

    logger = CSVLogger("logs", name=args.name)
    
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, progress_bar_refresh_rate=args.refresh_rate, **config.lightning.trainer)
    #trainer = pl.Trainer(callbacks=callbacks, logger=logger, **config.lightning.trainer)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
