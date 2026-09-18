from pathlib import Path

from lightning.pytorch.cli import LightningCLI


class SamsoneCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--current-time", type=str)
        parser.add_argument("--experiment-name", type=str)
        parser.add_argument("--hf-text-model-name", type=str)
        parser.add_argument("--hf-audio-model-name", type=str)
        parser.add_argument("--sample_rate", type=int)
        parser.add_argument("--experiment-state-load-path", type=str, default=None)
        parser.add_argument("--optimizer", type=str, default=None)
        parser.add_argument("--lr_scheduler", type=str, default=None)

    def before_instantiate_classes(self):
        if not (
            Path(self.config.trainer.logger.init_args.save_dir).parent
            / self.config.current_time
        ).is_dir():
            self.config.experiment_name = (
                f"{self.config.experiment_name}_{self.config.current_time}"
            )
            self.config.trainer.logger.init_args.save_dir = (
                Path(self.config.trainer.logger.init_args.save_dir)
                / self.config.current_time
            )
