from model_builder import ViT
import torch
from torchvision import transforms
from model_builder import IMG_SIZE
from model_builder import ViT
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from typing import List

# device = "cuda" if torch.cuda.is_available() else "cpu"

def pred_and_plot_image(model: torch.nn.Module, 
                        image_path: str, 
                        class_names: List[str] = None, 
                        transform=None,
                        device: torch.device = "cpu"):
    """Makes a prediction on a target image and plots the image with its prediction."""
    
    # 1. Load in image and convert the tensor values to float32
    target_image = Image.open(image_path) 
    
    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    # target_image = target_image / 255. 
    
    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)
    
    # 4. Make sure the model is on the target device
    model.to(device)
    
    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
    
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))
        
    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else: 
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show();

def predict_image(model, img_path):

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        ]) 

    # Initialize the model
    vit = ViT(num_classes=3)
    # Load trained model
    vit.load_state_dict(torch.load(model, map_location=torch.device("cpu")))
    # Predict and show the results
    pred_and_plot_image(model=vit, image_path=img_path, class_names=["pizza", "steak", "sushi"], transform=test_transform)


def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default="./models/vit_pretrained_3cls.pt", help="model path")
    
    # parser.add_argument("--img_path", type=str, default="./data/pizza_steak_sushi/test/pizza/195160.jpg", help="image path")
    # parser.add_argument("--img_path", type=str, default="./data/pizza_steak_sushi/test/steak/1868005.jpg", help="image path")
    parser.add_argument("--img_path", type=str, default="./data/pizza_steak_sushi/test/sushi/1600999.jpg", help="image path")
    
    opt = parser.parse_args()
    
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    predict_image(**vars(opt))