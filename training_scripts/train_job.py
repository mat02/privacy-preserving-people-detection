import os
import torch
os.environ["OMP_NUM_THREADS"]='8'
from pathlib import Path

from ultralytics import YOLO
from scrambled_train_utils import get_modified_state_dict

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Sample script for parsing command-line arguments.")

    # Required arguments
    parser.add_argument('--prefix', type=str, required=True, help='Prefix string')
    parser.add_argument('--suffix', type=str, required=False, default='', help='Suffix string')
    parser.add_argument('--device_id', type=int, choices=range(0, 1000), help='Device ID (0 or bigger)')
    parser.add_argument('--model_cfg_name', type=str, required=True, help='Model configuration name')
    parser.add_argument('--dataset', type=str, nargs=2, metavar=('NAME', 'PATH'), required=True,
                        help='Dataset as a pair of name and path')
    parser.add_argument('--batch_size', type=int, choices=range(1, 1000), required=True,
                        help='Batch size (1 or bigger)')
    
    # Optional argument with a conditional type (False or an integer)
    parser.add_argument('--patience', type=lambda x: False if x.lower() == 'false' else int(x),
                        default=False, help='Patience value (False or an integer)')

    # Required integer argument
    parser.add_argument('--seed', type=int, required=True, help='Seed integer')

    # Catch arbitrary key-value pairs
    parser.add_argument('--extra', nargs='*', metavar='KEY=VALUE', help='Additional key-value pairs')

    args = parser.parse_args()

    # Parse extra key-value arguments
    if args.extra:
        extra_args = {}
        for item in args.extra:
            key, value = item.split('=', 1)
            try:
                if value in ['True', 'False']:
                    value = value == 'True'
                else:
                    raise ValueError()
            except ValueError:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as a string if it cannot be converted to float
            extra_args[key] = value
        args.extra = extra_args

    return args

def run(args):
    if ".pt" not in args.model_cfg_name:
        base_model_name = args.model_cfg_name[:-5]
    else:
        base_model_name = Path(args.model_cfg_name)
        if "weights" in str(base_model_name):
            base_model_name = base_model_name.parent.parent.name
        base_model_name = str(base_model_name)[:-3]
    
    project = f"{args.dataset[0]}_{args.prefix}"
    exp_name = f"{base_model_name}{args.suffix}"

    model = YOLO(args.model_cfg_name)
    
    if "base_for_scrambling" in args.extra:
        pretrained_state_dict = get_modified_state_dict(args.extra["base_for_scrambling"], 3, 1)
        model.model.load_state_dict(pretrained_state_dict, strict=False)
        args.extra["freeze_layers"] = ['model.0.', 'model.3.', 'model.4.']
        
        del args.extra["base_for_scrambling"]

    model.train(
        data=args.dataset[1],
        batch=args.batch_size,
        device=[args.device_id],
        patience=args.patience,
        seed=args.seed,
        name=exp_name,
        project=project,
        deterministic=True,
        **args.extra
    )
    

if __name__ == "__main__":
    args = parse_args()

    # Display parsed arguments
    print("Parsed Arguments:")
    print(f"Prefix: {args.prefix}")
    print(f"Device ID: {args.device_id}")
    print(f"Model Config Name: {args.model_cfg_name}")
    print(f"Dataset Name: {args.dataset[0]}, Path: {args.dataset[1]}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Patience: {args.patience}")
    print(f"Seed: {args.seed}")
    if args.extra:
        print("Additional Key-Value Pairs:")
        for key, value in args.extra.items():
            print(f"  {key}: {value}")
    else:
        print("No additional key-value pairs provided.")
        
    run(args)
    