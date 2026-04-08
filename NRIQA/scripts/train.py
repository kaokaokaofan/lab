from torch.utils.data import DataLoader

from nriqa.config import DATA, MODEL, TRAIN
from nriqa.data.dataset import MRIQADataset
from nriqa.models.backbone import build_backbone, get_device
from nriqa.trainer import RegressorTrainer, create_criterion, create_optimizer, save_checkpoint


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
        batch_size=TRAIN.batch_size,
        shuffle=True,
        num_workers=TRAIN.num_workers,
        pin_memory=TRAIN.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN.batch_size,
        shuffle=False,
        num_workers=TRAIN.num_workers,
        pin_memory=TRAIN.pin_memory,
    )

    model = build_backbone(MODEL.backbone, MODEL.use_pretrained, MODEL.num_outputs)
    optimizer = create_optimizer(model, TRAIN.optimizer, TRAIN.lr, TRAIN.momentum)
    criterion = create_criterion(TRAIN.criterion)
    trainer = RegressorTrainer(model, optimizer, criterion, device=device)

    best_rmse = float("inf")
    for epoch in range(TRAIN.epochs):
        train_loss = trainer.train_one_epoch(train_loader)
        result = trainer.evaluate(test_loader)

        print(
            f"Epoch [{epoch + 1}/{TRAIN.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"test_loss={result['loss']:.4f} "
            f"RMSE={result['rmse']:.4f} "
            f"PLCC={result['plcc']:.4f} "
            f"SRCC={result['srcc']:.4f}"
        )

        if result["rmse"] < best_rmse:
            best_rmse = result["rmse"]
            best_path = save_checkpoint(trainer.model, TRAIN.save_dir, f"{TRAIN.save_name}_best")
            print("best model saved to", best_path)

    final_path = save_checkpoint(trainer.model, TRAIN.save_dir, f"{TRAIN.save_name}_final")
    print("final model saved to", final_path)


if __name__ == "__main__":
    main()
