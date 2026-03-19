"""Tests for AudioSetDataModule."""

from audioset_classification.data.data_module import AudioSetDataModule


def test_audioset_data_module_fit(mock_manifests_dir, mock_features_dir):
    """DataModule setup('fit') creates train and val loaders."""
    dm = AudioSetDataModule(
        manifests_dir=str(mock_manifests_dir),
        features_dir=str(mock_features_dir),
        num_classes=3,
        batch_size=2,
        num_workers=0,
        synthetic=True,
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    x, y = batch
    assert x.shape[0] <= 2
    assert x.shape[1] == 128
    assert y.shape[1] == 3

    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    vx, vy = val_batch
    assert vx.shape[1] == 128
    assert vy.shape[1] == 3


def test_audioset_data_module_test(mock_manifests_dir, mock_features_dir):
    """DataModule setup('test') creates a test loader."""
    dm = AudioSetDataModule(
        manifests_dir=str(mock_manifests_dir),
        features_dir=str(mock_features_dir),
        num_classes=3,
        batch_size=2,
        num_workers=0,
        synthetic=True,
    )
    dm.setup("test")
    test_loader = dm.test_dataloader()
    batch = next(iter(test_loader))
    x, y = batch
    assert x.shape[1] == 128
    assert y.shape[1] == 3
