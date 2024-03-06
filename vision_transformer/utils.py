from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter



def create_writer(experiment_name="classification",
                  model_name="vit",
                  extra="30_epochs"):
    """Create a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log dir.
    Example Usage:
    # Create a writer saving to "runs/2024-02-03_classification_vit_10_epochs"
    writer = create_writer(experiment_name="classification",
                            model_name="vit",
                            extra="10_epochs")
    
    # The above is the same as:
    writer = SummaryWriter(log_dir="runs/2024-02-03_classification_vit_10_epochs")
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")  # Return current time
    file_name = "_".join([timestamp, experiment_name, model_name, extra])
    log_dir = os.path.join("runs", file_name)
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir} ...")
    return SummaryWriter(log_dir=log_dir)




    
    
    