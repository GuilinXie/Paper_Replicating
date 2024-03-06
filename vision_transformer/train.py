from model_builder import ViT
import torch
from torchvision import transforms
from model_builder import IMG_SIZE, PATCH_SIZE, HIDDEN_D, NUM_HEADS
# from data_setup import create_dataloaders
from data_setup import ImageFolderCustom, download_data
from torch.utils.data import DataLoader
import os
from model_builder import ViT
from engine import train
from utils import create_writer


# Hyperparameters
BATCH_SIZE = 16
NW = min([1, os.cpu_count(), BATCH_SIZE if BATCH_SIZE >= 1 else 0, 8])
EPOCHS = 30
IMG_SIZE=224

device = "cuda" if torch.cuda.is_available() else "cpu"


image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")
train_dir = image_path / "train"
test_dir = image_path / "test"

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) 

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) 


train_data = ImageFolderCustom(targ_dir=train_dir, transform=train_transform)
test_data = ImageFolderCustom(targ_dir=test_dir, transform=test_transform)

train_loader = DataLoader(dataset=train_data,
                          batch_size=BATCH_SIZE,
                          num_workers=NW,
                          shuffle=True)

test_loader = DataLoader(dataset=test_data,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False)

# Instantiate an instance of the ViT model
torch.manual_seed(42)
vit = ViT(num_classes=3)
vit = vit.to(device=device)

# Setup the optimizer to optimizer out ViT model parameters using hyperparameters above
optimizer = torch.optim.Adam(vit.parameters(), 
                            lr=3e-3,                    # Base lr for ViT-* ImageNet-1k
                            betas=(0.9, 0.999),         # Default values but also mentioned in ViT paper section 4.1 (training & fine-tuning)       
                            weight_decay=0.3            # From the ViT paper section 4.1 training & fine-tuning
                            )

# Setup the loss function for multi-class classification
loss_fn = torch.nn.CrossEntropyLoss()

results = train(model=vit,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                optimzier=optimizer,
                loss_fn=loss_fn,
                epochs=EPOCHS,
                device=device,
                writer=create_writer(experiment_name="cls", model_name="vit", extra="30_epochs"))


## Pretrained - transfer learning
# # 1. Get pretrained weights for ViT-Base
# pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT # requires torchvision >= 0.13, "DEFAULT" means best available

# # 2. Setup a ViT model instance with pretrained weights
# pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# # 3. Freeze the base parameters
# for parameter in pretrained_vit.parameters():
#     parameter.requires_grad = False

# # 4. Change the classifier head (set the seeds to ensure same initialization with linear head)
# set_seeds()
# pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
# # pretrained_vit 
