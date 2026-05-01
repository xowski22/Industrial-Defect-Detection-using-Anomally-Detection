from pathlib import Path

from anomalib.loggers import AnomalibWandbLogger
from anomalib.engine import Engine


def build_anomalib_engine(
    exp_dir: Path,
    model_name: str,
    category: str,
    project: str = "anomaly-detection-mvtec",
) -> Engine:
    """
    Build an Anomalib Engine with a W&B logger attached.

    Args:
        exp_dir: Directory where checkpoints and logs will be saved.
        model_name: Name of the model (e.g. 'patchcore').
        category: MVTec category being trained on.
        project: W&B project name.

    Returns:
        Configured Anomalib Engine instance.
    """
    logger = AnomalibWandbLogger(
        project=project,
        name=f"{model_name}_{category}",
        group=category,
        tags=[model_name, category],
        log_model=False,
    )

    engine = Engine(
        max_epochs=1,
        devices=1,
        accelerator="auto",
        default_root_dir=str(exp_dir),
        logger=logger,
    )

    return engine
