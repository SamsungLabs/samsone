from datetime import datetime
import logging
from pathlib import Path

from clearml import Task
from clearml.backend_api.session.defs import MissingConfigError
from omegaconf import OmegaConf
from setproctitle import setproctitle
from torch import set_float32_matmul_precision

from samsone import SamsoneCLI

logger = logging.getLogger("lightning")


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")


OmegaConf.register_new_resolver("now", get_current_time)


if __name__ == "__main__":
    cli = SamsoneCLI(
        run=False,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        auto_configure_optimizers=False,
    )

    # Access experiment_name from the parsed arguments
    experiment_name = cli.config.get("experiment_name", "default_experiment")
    try:
        Task.init(project_name="samsone", task_name=experiment_name)
    except MissingConfigError:
        logger.warning(
            "ClearML is not configured, running without it "
            "(set CLEARML_OFFLINE_MODE=1 to keep an offline ClearML log)."
        )

    """
    If the output dir already exists and has a "last.ckpt" checkpoint,
    load this checkpoint and continue training.
    """
    output_dir = Path(cli.trainer.logger.save_dir)
    setproctitle(output_dir.parent.name)

    last_ckpt_path = output_dir / "checkpoints" / "last.ckpt"
    ckpt_load_path = None

    if (
        cli.config.experiment_state_load_path
        and Path(cli.config.experiment_state_load_path).is_file()
    ):
        ckpt_load_path = cli.config.experiment_state_load_path
    elif last_ckpt_path.is_file():
        logger.info(f"Loading existing training state from: {last_ckpt_path}")
        ckpt_load_path = last_ckpt_path

    if cli.trainer.precision == "32-true":
        set_float32_matmul_precision("high")
    elif cli.trainer.precision in {"16-mixed", "bf16-mixed"}:
        set_float32_matmul_precision("medium")

    cli.trainer.fit(
        model=cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_load_path
    )
    cli.trainer.test(model=cli.model, datamodule=cli.datamodule)

    valid_best_ckpt_path = next(
        Path(output_dir / "checkpoints").glob("val_best*"), None
    )

    if valid_best_ckpt_path:
        cli.trainer.test(
            model=cli.model, datamodule=cli.datamodule, ckpt_path=valid_best_ckpt_path
        )
