import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 pixels
    transforms.ToTensor(),  # Convert to PyTorch Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization
])

# Video dataset class: a single video dataset would consist 9 sequence of frames with highlighted skeleton points and
# white/original background
class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.videos = []
        curr_vid_id = '0'
        curr_vid_list = []
        for c in self.classes:
            # somehow it reads the Mac .DS_store file...
            if c.startswith('.'):
                continue

            pro_am, vid_id, swing_pos, confidence_score, _ = c.split('_')

            if vid_id == curr_vid_id:
                curr_vid_list.append((vid_id, swing_pos, float(confidence_score), pro_am, os.path.join(root_dir, c)))
            else:
                if len(curr_vid_list) != 0:
                    self.videos.append(list(curr_vid_list))
                curr_vid_id = vid_id
                curr_vid_list.clear()
                curr_vid_list.append((vid_id, swing_pos, float(confidence_score), pro_am, os.path.join(root_dir, c)))

        if len(curr_vid_list) != 0:
            self.videos.append(list(curr_vid_list))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        frames = []
        for frame in self.videos[idx]:
            vid_id, swing_position, confidence_score, pro_am, img_path = frame
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Stack frames into one tensor
        # TODO: figure out a way to encode swing_position & confidence score
        video = torch.hstack(frames)
        label = 1 if (pro_am == 'pro') else 0
        return video, label


# Define custom collate_fn to handle variable number of frames
# def collate_fn(batch=16):
#     videos = []
#     labels = []
#     for video, label in batch:
#         videos.append(video)
#         labels.append(label)
#
#     return torch.tensor(videos), torch.tensor(labels)


# trial_root_dir = 'dataset/test_dataset'
# currently using 3D skeleton points w/ white background as training examples
positive_dir = 'dataset/pro_swing-position_skeleton_original-background_frames'
negative_dir = 'dataset/amateur_swing-position_skeleton_white-background_frames'
positive_dataset = VideoDataset(positive_dir, transform)
negative_dataset = VideoDataset(negative_dir, transform)
full_dataset = torch.utils.data.ConcatDataset([positive_dataset, negative_dataset])

# Split the data into training and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# Load pre-trained model
model = models.resnet50(pretrained=True)

# Freeze the model parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer for binary classification
model.fc = nn.Linear(model.fc.in_features, 1)

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


# Access batch
# TODO: implement batch later
# for batch in dataloader:
#     videos, labels = batch
#     # each item in 'videos' is a tensor with all frames of a single video

best_dev_acc = 0.0

# Train the model
for epoch in range(10):  # for 10 epochs

    # Train mode
    model.train()
    train_loss = 0.0
    for inputs, labels in tqdm(train_dataloader, desc=f'train-{epoch}', disable=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), labels.float())

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print('Epoch: ', epoch, 'Training Loss: ', train_loss)
    print('Finishing Training the model. Now starting to evaluate...')

    # Validate the model
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader, desc=f'validation', disable=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int().squeeze()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    print('Dev Set Acc: ', acc)
    print('Dev Set F1 score: ', f1)

    if acc > best_dev_acc:
        best_dev_acc = acc
        # Save the trained model
        torch.save(model.state_dict(), 'pro_am_classifier.pth')

print('Overall Best Dev Acc: ', best_dev_acc)

# TODO: implement test set