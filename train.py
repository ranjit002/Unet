import torch
from tqdm import tqdm

from architecture import UNET, BCELogitsWeight, DiceLoss
from preprocess import train_dataloader, val_dataloader


def train(model_, optim, epochs, loss_func, device_):
    model_.to(device_)

    for _ in tqdm(range(epochs)):
        model_.train()
        train_running_loss = 0
        train_running_dc = 0

        for img_mask in tqdm(train_dataloader, position=0, leave=True):
            img = img_mask[0].float().unsqueeze(1).to(device_)
            mask = img_mask[1].float().unsqueeze(1).to(device_)

            y_pred = model_(img)
            optim.zero_grad(set_to_none=True)

            dc = dice_coefficient(y_pred, mask)
            loss = loss_func(y_pred, mask)

            train_running_loss += loss.item()
            train_running_dc += dc.item()

            loss.backward()
            optim.step()

        train_loss = train_running_loss / (len(train_dataloader) + 1)
        train_dc = train_running_dc / (len(train_dataloader) + 1)

        val_running_loss = 0
        val_running_dc = 0

        model_.eval()
        with torch.no_grad():
            for img_mask in tqdm(val_dataloader, position=0, leave=True):
                img = img_mask[0].float().unsqueeze(1).to(device_)
                mask = img_mask[1].float().unsqueeze(1).to(device_)

                y_pred = model_(img)
                loss = loss_func(y_pred, mask)
                dc = dice_coefficient(y_pred, mask)

                val_running_loss += loss.item()
                val_running_dc += dc.item()

            val_loss = val_running_loss / (len(val_dataloader) + 1)
            val_dc = val_running_dc / (len(val_dataloader) + 1)

        print("-" * 30)
        print("TRAINING")
        print(f"Loss: {train_loss:.4f}")
        print(f"DICE accuracy: {train_dc}")
        print("-" * 30)
        print("VALIDATION")
        print(f"Loss: {val_loss:.4f}")
        print(f"DICE accuracy: {val_dc}")
        print("-" * 30)


def dice_coefficient(pred, target):
    smooth = 1e-4
    iflat = torch.sigmoid(pred.clone()).contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()
    return (2.0 * intersection + smooth) / (union + smooth)


if __name__ == "__main__":
    # Turn off debugging settings
    torch.autograd.set_detect_anomaly(False, False)
    torch.autograd.profiler.profile(False)
    torch.set_float32_matmul_precision("medium")

    model = UNET(1, 1, features=[64, 128])
    model.load_state_dict(torch.load("models/model.pth"))
    loss_fn = DiceLoss()

    LEARNING_RATE = 1e-7
    EPOCHS = 5

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )

    print(f"lr = {LEARNING_RATE}")
    print(type(loss_fn))

    train(model, optimizer, EPOCHS, loss_fn, device)

    # Saving the model
    torch.save(model.state_dict(), "models/model.pth")
