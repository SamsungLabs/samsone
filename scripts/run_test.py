import logging
from pathlib import Path

from clearml import Task
from setproctitle import setproctitle
from torch import set_float32_matmul_precision

from samsone import SamsoneCLI

logger = logging.getLogger("lightning")


if __name__ == "__main__":
    cli = SamsoneCLI(
        run=False,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        auto_configure_optimizers=False,
    )

    # Access experiment_name from the parsed arguments
    experiment_name = cli.config.get("experiment_name", "default_experiment")
    Task.init(project_name="samsone", task_name=experiment_name)

    output_dir = Path(cli.trainer.logger.save_dir)
    setproctitle(f"{output_dir.parent.name}-test")

    if cli.trainer.precision == "32-true":
        set_float32_matmul_precision("high")
    elif cli.trainer.precision in {"16-mixed", "bf16-mixed"}:
        set_float32_matmul_precision("medium")

    cli.trainer.test(
        model=cli.model,
        datamodule=cli.datamodule,
        ckpt_path=cli.config.experiment_state_load_path,
    )
