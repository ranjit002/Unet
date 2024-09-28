import torch
from model import UNET
from torch import nn
from tqdm import tqdm  # progress bar


class BCELogitsWeight(nn.Module):
    def __init__(self, alpha=0.02):
        """
        Binary Cross Entropy, with a rescaling of the cross entropies
        so that the two classes are proportionally represented.

        Args:
            alpha (float, optional): Fraction of classes = 1 in the target mask. Defaults to 0.02.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, output, target):
        """
        Args:
            output (tensor): prediction of model WITHOUT sigmoid applied (Logits)
            target (tensor): desired output of model AFTER sigmoid (mask)
        """
        assert output.shape == target.shape, "Prediction and target shape mismatch"

        pred = output.sigmoid().view(-1)  # Apply sigmoid and flatten tensor
        target_flat = target.view(-1)

        epsilon = 1e-6
        pred = pred.clamp(
            epsilon, 1 - epsilon
        )  # Prevent log(pred) and log(1 - pred) overflow

        return (
            -1
            * (
                (1 - self.alpha) * target_flat * pred.log()
                + self.alpha * (1 - target_flat) * (1 - pred).log()
            ).mean()
        )


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-7):
        """
        Dice loss with smoothing

        Args:
            smooth (float): smoothing for preventing zero division
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, output, target):
        """
        Args:
            output (tensor): prediction of model BEFORE sigmoid is applied
            target (tensor): desired output of the model AFTER sigmoid is applied (i.e the mask)
        """
        assert output.shape == target.shape, "Prediction and target shape mismatch"

        pred = output.sigmoid().view(-1)  # Apply sigmoid and flatten tensor
        target_flat = target.view(-1)

        intersection = (pred * target_flat).sum()
        union = pred.sum() + target_flat.sum()

        return 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)


def dice_accuracy(pred, target, smooth=1e-4):
    """
    Implementation of dice accuracy

    Args:
        pred (tensor): Prediction of model BEFORE sigmoid
        target (tensor): Target output of model
        smooth (float, optional): smoothing for preventing zero division. Defaults to 1e-4.

    Returns:
        float: dice accuracy
    """
    # Flattened predictions
    pflat = torch.sigmoid(pred.clone()).contiguous().view(-1)
    # Flattened targets
    tflat = target.contiguous().view(-1)

    intersection = (pflat * tflat).sum()
    union = pflat.sum() + tflat.sum()

    return (2.0 * intersection + smooth) / (union + smooth)


def train(model, train_data, val_data, optim, epochs, loss_func, dvc):
    """
    Train PyTorch model

    Args:
        model (torch.Module): model to be trained
        train_data (_type_): _description_
        val_data (_type_): _description_
        optim (torch optimizer): optimizer to be used to train model
        epochs  (int): num of epochs to train model for
        loss_func (func): function to evaluate model performance
        dvc (torch.device): device to compute training on
    """
    model.to(dvc)
    print(f"Learning Rate = {optim.param_groups[0]['lr']}")

    for _ in tqdm(range(epochs)):
        train_running_loss = 0
        train_running_dc = 0

        model.train()
        for idx, img_mask in enumerate(tqdm(train_data, position=0, leave=True)):
            img = img_mask[0].float().unsqueeze(1).to(dvc)
            mask = img_mask[1].float().unsqueeze(1).to(dvc)

            y_pred = model(img)
            optim.zero_grad()  # Reset gradients to zero

            dc = dice_accuracy(y_pred, mask)
            loss = loss_func(y_pred, mask)

            train_running_loss += loss.item()
            train_running_dc += dc.item()

            loss.backward()  # Compute gradients
            optim.step()  # Update weights

        train_loss = train_running_loss / (idx + 1)
        train_dc = train_running_dc / (idx + 1)

        val_running_loss = 0
        val_running_dc = 0

        model.eval()
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_data, position=0, leave=True)):
                img = img_mask[0].float().unsqueeze(1).to(dvc)
                mask = img_mask[1].float().unsqueeze(1).to(dvc)

                y_pred = model(img)
                loss = loss_func(y_pred, mask)
                dc = dice_accuracy(y_pred, mask)

                val_running_loss += loss.item()
                val_running_dc += dc.item()

            val_loss = val_running_loss / (idx + 1)
            val_dc = val_running_dc / (idx + 1)

        print("-" * 30)

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training DICE accuracy: {train_dc}")

        print("\n")

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation DICE accuracy: {val_dc}")

        print("-" * 30)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    CNN_model = UNET(1, 1, layer_sizes=[64, 128, 256])

    LEARNING_RATE = 1e-5
    loss_fn = DiceLoss()
    optimizer = torch.optim.AdamW(
        CNN_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5
    )
    EPOCHS = 5

    #! NEED to parse trainig data
    train_dl, valid_dl = None, None

    train(
        CNN_model,
        optimizer,
        train_dl,
        valid_dl,
        epochs=EPOCHS,
        loss_func=loss_fn,
        dvc=device,
    )

    # Saving the model
    torch.save(CNN_model.state_dict(), "model.pth")
