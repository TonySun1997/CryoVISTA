# Configuration defaults (safe on import)

import torch

train_dataset_path = "train_dataset/*/"
test_dataset_path = "test_dataset"
my_dataset_path = "my_dataset"
output_path = "output"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pin_memory = False
num_workers = 8

num_channels = 1
num_classes = 1
num_levels = 3

learning_rate = 1e-4
num_epochs = 200
batch_size = 6

input_image_width = 1024
input_image_height = 1024
input_shape = 1024

logging = False

cryosegnet_checkpoint = "pretrained_models/cryosegnet.pth"
sam_checkpoint = "pretrained_models/sam_vit_h_4b8939.pth"
model_type = "vit_h"

empiar_id = 10081
file_name = "10081.star"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CryoSegNet Training")
    parser.add_argument("--train_dataset_path", type=str, default=train_dataset_path, help="Path to the training dataset")
    parser.add_argument("--test_dataset_path", type=str, default=test_dataset_path, help="Path to the test dataset")
    parser.add_argument("--my_dataset_path", type=str, default=my_dataset_path, help="Path to your own dataset")
    parser.add_argument("--output_path", type=str, default=output_path, help="Output directory")

    parser.add_argument("--device", type=str, default=device, help="Device (cuda:0 or cpu)")
    parser.add_argument("--pin_memory", action="store_true", help="Enable pin_memory for data loading if using CUDA")
    parser.add_argument("--num_workers", type=int, default=num_workers, help="Number of data loading workers")

    parser.add_argument("--num_channels", type=int, default=num_channels, help="Number of input channels")
    parser.add_argument("--num_classes", type=int, default=num_classes, help="Number of classes")
    parser.add_argument("--num_levels", type=int, default=num_levels, help="Number of levels in the model")

    parser.add_argument("--learning_rate", type=float, default=learning_rate, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=num_epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=batch_size, help="Batch size")

    parser.add_argument("--input_image_width", type=int, default=input_image_width, help="Input image width")
    parser.add_argument("--input_image_height", type=int, default=input_image_height, help="Input image height")
    parser.add_argument("--input_shape", type=int, default=input_shape, help="Input image shape")

    parser.add_argument("--logging", action="store_true", help="Enable logging")

    parser.add_argument("--cryosegnet_checkpoint", type=str, default=cryosegnet_checkpoint, help="Path to CryoSegNet checkpoint")
    parser.add_argument("--sam_checkpoint", type=str, default=sam_checkpoint, help="Path to SAM checkpoint")
    parser.add_argument("--model_type", type=str, default=model_type, help="SAM Model type")

    parser.add_argument("--empiar_id", type=int, default=10081, help="EMPIAR ID for prediction")
    parser.add_argument("--file_name", type=str, default="10081.star", help="Filename for picked proteins coordinates")

    args = parser.parse_args()

    
    globals().update({k: getattr(args, k) for k in vars(args)})

    
    print("CryoSegNet Training Configuration (standalone config.py)")
    for k in vars(args):
        print(f"{k}: {getattr(args, k)}")
