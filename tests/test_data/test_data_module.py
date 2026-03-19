"""Tests for DataModule."""

from audioset_classification.data.data_module import AudioSetDataModule


def test_audioset_data_module_setup(mock_audioset_dir):
    """DataModule setup creates train/val loaders."""
    dm = AudioSetDataModule(
        data_dir=str(mock_audioset_dir),
        ontology_path=str(mock_audioset_dir / "class_labels_indices.csv"),
        split="balanced_train",
        batch_size=2,
        max_segments=10,
        synthetic=True,
        num_classes=3,
        num_workers=0,
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    x, y = batch
    assert x.shape[0] <= 2
    assert x.shape[1] == 128
    assert y.shape[1] == 3
