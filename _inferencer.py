import os
import argparse
import torch
from model import ChordConditionedMelodyTransformer
from dataset import collate_fn
from utils.utils import logger

class CMTinferencer:
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device

        # Initialize the model
        self.model = ChordConditionedMelodyTransformer()  # Assuming your model doesn't require additional arguments

        # Load the trained model
        self.load_model()

        # Set the model to evaluation mode
        self.model.eval()

    def load_model(self):
        # Load the trained model weights
        if os.path.isfile(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            logger.info("Loaded trained model weights from: {}".format(self.model_path))
        else:
            raise FileNotFoundError("Model checkpoint not found at: {}".format(self.model_path))

    def inference(self, data):
        # Perform inference on new data
        with torch.no_grad():
            # Transfer data to device
            data = {key: value.to(self.device) for key, value in data.items()}

            # Forward pass
            result_dict = self.model(data['rhythm'], data['pitch'][:, :-1], data['chord'], False, rhythm_only=False)

            # Process the inference results as needed
            # For example, you can return the output pitches
            return result_dict['pitch'].cpu().numpy()

def main(args):
    # Initialize the inferencer
    inferencer = CMTinferencer(args.model_path, args.device)

    # Load your data here, assuming `data` is your data dictionary
    # data = ...

    # Perform inference
    result = inferencer.inference(data)
    print("Inference result:", result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMT Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint (.pth.tar)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (default: cpu)")
    # Add more arguments as needed

    args = parser.parse_args()
    main(args)
