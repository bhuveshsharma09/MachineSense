"""Stage 3B path scheme: variant, seed and fold are part of every output path and run name."""
from ml.paths import RunPaths, artifacts_dir, results_dir

LEGACY_SEED42 = [results_dir() / "scores" / f"{v}_dev{s}" for v in ("cnn7", "cnn6")
                 for s in (".parquet", "_embeddings.npz")] + \
                [results_dir() / "cnn" / f"{v}_dev_train.json" for v in ("cnn7", "cnn6")] + \
                [artifacts_dir() / "cnn" / f"{v}_dev" / f"fold{k}.pt" for v in ("cnn7", "cnn6") for k in range(3)]


def test_runs_differing_only_in_seed_write_to_different_paths():
    a, b = RunPaths("cnn", "cnn7", 43), RunPaths("cnn", "cnn7", 44)
    pa, pb = set(a.all_paths()), set(b.all_paths())
    assert len(pa) == len(a.all_paths()) and not pa & pb
    assert a.run_name(0) != b.run_name(0) and a.tags(1)["seed"] == "43" and a.tags(1)["fold"] == "1"
    assert all("s43" in str(p) for p in pa) and all("s44" in str(p) for p in pb)


def test_variant_and_fold_are_in_the_paths():
    a, b = RunPaths("ae", "ae7", 42), RunPaths("ae", "ae6", 42)
    assert not set(a.all_paths()) & set(b.all_paths())
    assert a.weights(0) != a.weights(1) and a.norm_stats(0) != a.norm_stats(2)
    assert a.run_name() == "ae7_s42_dev" and a.run_name(2) == "ae7_s42_dev_fold2"


def test_new_scheme_never_touches_the_stage_3a_seed42_outputs():
    for v in ("cnn7", "cnn6"):
        assert not set(RunPaths("cnn", v, 42).all_paths()) & set(LEGACY_SEED42)


def test_withheld_class_is_in_the_paths():
    a = RunPaths("loco", "cnn6", 42, protocol="loco", withheld="imbalance")
    b = RunPaths("loco", "cnn6", 42, protocol="loco", withheld="overhang_ball")
    assert not set(a.all_paths()) & set(b.all_paths())
    assert all("imbalance" in str(p) for p in a.all_paths())
    assert a.tags(1)["withheld"] == "imbalance" and "imbalance" in a.run_name(1)
    assert "withheld" not in RunPaths("cnn", "cnn6", 42).tags()
