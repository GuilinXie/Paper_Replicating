import torch
from tqdm.auto import tqdm

def train_step(model,
               dataloader,
               loss_fn,
               optimizer,
               device
               ):
    """Train a model for a single epoch.
    Args:
        model: A model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A loss functions to minimize.
        optimizer: An optimizer to help minimize the loss function.
        device: A target device to compute on(eg: "cuda" or "cpu")
        
    Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:
        (0.112, 0.8743)  
    """
    
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through dataloader
    for batch, (X, y) in enumerate(dataloader):
        
        # Send data to target device
        # if batch % 2 == 0:
        #     print(f"train: {batch} / {len(dataloader)}")
        X, y = X.to(device), y.to(device)
        
        # 1. Forward Pass
        y_pred = model(X)
        
        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        
        # 4. Loss backward
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model,
              dataloader,
              loss_fn,
              device):
    """Tests a model on a testing dataset.

    Args:
        model: A model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A loss function to calculate loss on the test data.
        device: A target device to compute on(eg: "cuda" or "cpu")
    
    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:
        (0.0223, 0.8985)
    """
    
    # Put model in eval mode for testing
    model.eval()
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred_logits = model(X)
            
            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model,
          train_dataloader,
          test_dataloader,
          optimzier,
          loss_fn,
          epochs,
          device,
          writer=None):
    """Train and Test a model

    Args:
        model: A model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimzier: An optimizer to help minimize the loss functions.
        loss_fn: A loss function to calculate loss on both datasets/
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on(eg: "cuda" or "cpu")
        
    Returns:
        A dictionary of training and testing loss and accuracy metrics.
        Each metric has a value in a list for each epoch.
        In the form:{
            train_loss: [...],
            train_acc: [...],
            test_loss: [...],
            test_acc: [...]
        }
        For example if training for epochs = 2:
        {
            train_loss: [2.0601, 1.0537],
            train_acc: [0.3945, 0.5678],
            test_loss: [1.2641, 1.5706],
            test_acc: [0.3400, 0.3829]
        }
    """
    
    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimzier,
            device=device
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        # Print out results in current epoch
        print(f"Epoch: {epoch + 1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")
        
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={
                                   "train_loss": train_loss,
                                   "test_loss": test_loss
                               },
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy",
                               tag_scalar_dict={
                                   "train_acc": train_acc,
                                   "test_acc": test_acc
                               },
                               global_step=epoch)
            writer.close()
        else:
            pass
    return results
    
    
            
        