"""mlflow_run logs config, hashes, seed and git sha — to a temp store, never the real mlruns.db."""
from mlflow.tracking import MlflowClient

from ml.config import Config, config_sha256
from ml.utils import mlflow_run


def test_dummy_run_logs_provenance(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlruns.db'}"
    cfg = Config(seed=7)

    with mlflow_run(cfg, run_name="dummy", experiment="test", tracking_uri=uri,
                    artifact_root=tmp_path / "artifacts") as run:
        import mlflow
        mlflow.log_metric("dummy_metric", 1.0)

    got = MlflowClient(tracking_uri=uri).get_run(run.info.run_id)
    assert got.data.params["seed"] == "7"
    assert got.data.params["preprocess.fs_hz"] == "50000"
    assert got.data.tags["config_sha256"] == config_sha256(cfg)
    assert got.data.tags["preprocess_sha256"] == config_sha256(cfg.preprocess)
    assert got.data.tags["git_sha"]
    assert got.data.metrics["dummy_metric"] == 1.0
    assert (tmp_path / "artifacts").exists()
