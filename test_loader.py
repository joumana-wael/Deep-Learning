
from src.data.loaders import create_classification_loaders

def main():
    train_loader, val_loader, class_names = (
        create_classification_loaders(
            batch_size=32,
            seed=42,
        )
    )

    print("\n===== DATASET CHECK =====")
    print("Number of classes:", len(class_names))
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Total images:", len(train_loader.dataset) + len(val_loader.dataset))

    print("\n===== BATCH CHECK =====")
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Training drop_last:", train_loader.drop_last)
    print("Validation drop_last:", val_loader.drop_last)

    print("\n===== FIRST BATCH =====")
    images, labels = next(iter(train_loader))
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)

if __name__ == "__main__":
    main()
