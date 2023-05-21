import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

USE_GPU = True
dtype = torch.float32 # We will be using float throughout this tutorial.

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss.
print_every = 100
print('using device:', device)



NUM_TRAIN = 49000

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
videos_train = dset.CIFAR10('./golfdb-master/data/videos_160', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(videos_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

videos_val = dset.CIFAR10('./golfdb-master/data/videos_160', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(videos_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

videos_test = dset.CIFAR10('./golfdb-master/data/videos_160', train=False, download=True, 
                            transform=transform)


def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

# # Golf DB code

# For each swing video, we first detect the 9 frames --> 9 pictures

# see code in golfdb-master/test_video.py

# # MMPose code

# For every picture, use 3D pose estimation to pinpoint skeleton points 
# (result can be a bunch of vector points), then we vstack them together

model = init_model(
    args.config,
    args.checkpoint,
    device=args.device,
    cfg_options=cfg_options)

# init visualizer
model.cfg.visualizer.radius = args.radius
model.cfg.visualizer.alpha = args.alpha
model.cfg.visualizer.line_width = args.thickness

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.set_dataset_meta(
    model.dataset_meta, skeleton_style=args.skeleton_style)

# inference a single 
video_swing_position_skeleton_points = []
for _ in range(9):
    batch_results = inference_topdown(model, args.img)
    frame_skeleton_points = merge_data_samples(batch_results)
    video_swing_position_skeleton_points.append(frame_skeleton_points)
    

# Given all the 9 indices point, flatten and combine them sequentially order
flatten_video_swing_position_points = flatten(video_swing_position_skeleton_points)

# After flattening, so one video is one vector, vstack training example together

# For label vector y, currently since we are all using professional player swing, 
# the TODO item is to collect more amateur swing exmaples 
# so we can balance out more dataset distribution



# We need to wrap `flatten` function in a module in order to stack it
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)
    

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))



def train(model, optimizer, epochs=1):

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(loader_val, model)
                print()




model = None
optimizer = None

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

out_1, out_2, out_3, out_4, num_classes = 16, 32, 64, 128, 10
in_1, in_2, in_3, in_4 = 3, 16, 32, 64
filter_1, filter_2, filter_3, filter_4 = 5, 3, 3, 3

conv1 = nn.Sequential(
    nn.Conv2d(in_1, out_1, filter_1, padding=2),
    nn.BatchNorm2d(out_1),
    nn.ReLU(),
    nn.MaxPool2d(2)
)

conv2 = nn.Sequential(
    nn.Conv2d(in_2, out_2, filter_2, padding=1),
    nn.BatchNorm2d(out_2),
    nn.ReLU(),
    nn.MaxPool2d(2)
)

conv3 = nn.Sequential(
    nn.Conv2d(in_3, out_3, filter_3, padding=1),
    nn.BatchNorm2d(out_3),       
    nn.ReLU(),                        
    nn.MaxPool2d(2)                     
)   

conv4 = nn.Sequential(
    nn.Conv2d(in_4, out_4, filter_4, padding=1),
    nn.BatchNorm2d(out_4),     
    nn.ReLU(),                        
    nn.MaxPool2d(2)                     
)    

# Input: 4 x 4 x 64
# move to fully connected layers
fc =  nn.Sequential(
    nn.Dropout(0.2, inplace=True),
    nn.Linear(out_4 * 2 * 2, num_classes)
)

model = nn.Sequential(
    conv1, conv2, conv3, conv4, Flatten(), fc
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

train(model, optimizer, epochs=10)