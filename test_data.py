from utils.builders import Builder

builder = Builder("./experiments/cy3d")
train_loader, val_loader = builder.train_loader, builder.val_loader
for data in train_loader:
    d = data
    break