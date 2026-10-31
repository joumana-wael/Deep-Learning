import torch


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    """Train a model for one complete epoch.

    Returns:
        Average training loss and training accuracy.
    """

    # Put the model in training mode
    model.train()

    # Variables for keeping track of results
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    # Go through all batches in the training dataset
    for images, labels in train_loader:

        # Move images and labels to GPU (or CPU)
        images = images.to(
            device,
            non_blocking=(device.type == "cuda"),
        )

        labels = labels.to(
            device,
            non_blocking=(device.type == "cuda"),
        )

        # Clear gradients from the previous batch
        optimizer.zero_grad()

        # Ask the model to make predictions
        outputs = model(images)

        # Calculate how wrong the predictions are
        loss = criterion(outputs, labels)

        # Calculate gradients
        loss.backward()

        # Update the model's weights
        optimizer.step()

        # Add this batch's loss to the total loss
        train_loss_sum += loss.item() * images.size(0)

        # Get the predicted class
        predictions = outputs.argmax(dim=1)

        # Count correct predictions
        train_correct += (
            predictions == labels
        ).sum().item()

        # Count total images
        train_total += labels.size(0)

    # Calculate average loss for the entire epoch
    train_loss = train_loss_sum / train_total

    # Calculate accuracy as a percentage
    train_accuracy = (
        100.0 * train_correct / train_total
    )

    return train_loss, train_accuracy