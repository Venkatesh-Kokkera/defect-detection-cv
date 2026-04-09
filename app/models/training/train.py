import torch
import mlflow
import argparse
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

DEFECT_CLASSES = [
    "scratch", "crack", "dent",
    "contamination", "corrosion", "pass"
]

def get_transforms():
    """Get train and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform

def build_model(num_classes: int):
    """Build EfficientNet-B4 model."""
    model = models.efficientnet_b4(pretrained=True)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )
    return model

def train(args):
    """Main training loop with MLflow tracking."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Training on: {device}")

    # Transforms
    train_transform, val_transform = get_transforms()

    # Datasets
    train_dataset = datasets.ImageFolder(
        args.train_dir,
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        args.val_dir,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Model
    model = build_model(len(DEFECT_CLASSES)).to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    # MLflow tracking
    mlflow.set_experiment("defect-detection-cv")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "model": "EfficientNet-B4"
        })

        best_val_acc = 0.0

        for epoch in range(args.epochs):
            # Training
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            train_acc = 100.0 * correct / total

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = 100.0 * val_correct / val_total

            print(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Train Acc: {train_acc:.2f}% "
                f"Val Acc: {val_acc:.2f}%"
            )

            mlflow.log_metrics({
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "train_loss": train_loss
            }, step=epoch)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    "weights/efficientnet_defect.pt"
                )
                print(f"Best model
