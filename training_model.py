import torch 
import torchvision
from torchvision.models import resnet50
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# Constants
CLASSES = 2  # Two classes: grass and flower
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
CHECKPOINT_PATH = 'resnet50_weights_best_acc.tar'
DATASET_ROOT = 'path/to/dataset/'

# Set up TensorBoard
log_dir = f'runs/grass_flower_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
writer = SummaryWriter(log_dir)

# Load the model
model = resnet50(pretrained=False)  # Load without pre-trained weights
model.fc = torch.nn.Linear(model.fc.in_features, CLASSES)  # binary classification

# Check CUDA 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the checkpoint
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights

# Set the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Prepare the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ResNet normalization
])

# Assuming you have your dataset organized in 'train' and 'val' folders
train_dataset = datasets.ImageFolder(root=f'{DATASET_ROOT}/train', transform=transform)
val_dataset = datasets.ImageFolder(root=f'{DATASET_ROOT}/val', transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

