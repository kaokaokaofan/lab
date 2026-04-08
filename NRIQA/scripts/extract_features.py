from torch.utils.data import DataLoader

from nriqa.config import DATA, MODEL, FEATURE, CHECKPOINT_PATH, FEATURE_DIR
from nriqa.data.dataset import MRIQADataset
from nriqa.feature_pipeline import extract_dataset_features, save_feature_dict
from nriqa.models.backbone import FeatureExtractor, get_device

WEIGHT_PATH = CHECKPOINT_PATH


def main():
    device = get_device()
    print("Device:", device)

    train_dataset = MRIQADataset(
        DATA.train_image_txt,
        DATA.train_mos_txt,
        DATA.train_root,
        use_pretrained_norm=MODEL.use_pretrained,
    )
    test_dataset = MRIQADataset(
        DATA.test_image_txt,
        DATA.test_mos_txt,
        DATA.test_root,
        use_pretrained_norm=MODEL.use_pretrained,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=FEATURE.batch_size,
        shuffle=False,
        num_workers=FEATURE.num_workers,
        pin_memory=FEATURE.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=FEATURE.batch_size,
        shuffle=False,
        num_workers=FEATURE.num_workers,
        pin_memory=FEATURE.pin_memory,
    )

    extractor = FeatureExtractor(
        backbone_name=MODEL.backbone,
        weight_path=WEIGHT_PATH,
        use_pretrained=MODEL.use_pretrained,
        feature_mode=MODEL.feature_mode,
        device=device,
    )

    print("num extracted layers:", len(extractor.layer_names))
    print("first layers:", extractor.layer_names[:10])

    train_features = extract_dataset_features(extractor, train_loader)
    test_features = extract_dataset_features(extractor, test_loader)

    save_root = FEATURE_DIR
    save_feature_dict(train_features, extractor.layer_names, rf"{save_root}\train")
    save_feature_dict(test_features, extractor.layer_names, rf"{save_root}\test")
    print("saved to", save_root)


if __name__ == "__main__":
    main()
