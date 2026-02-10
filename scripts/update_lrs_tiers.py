#!/usr/bin/env python3
"""
Script to add xxs and xxl tiers to existing LRS configurations
Part of Tier 1 Performance Enhancements
"""

import json
import os

def add_enhanced_tiers(config_data):
    """Add xxs (1-5 images) and xxl (100+ images) tiers to each model"""
    
    for model_hash, model_config in config_data["data"].items():
        if isinstance(model_config, dict) and "xs" in model_config:
            # Skip if already has xxs and xxl
            if "xxs" in model_config and "xxl" in model_config:
                continue
                
            xs_config = model_config.get("xs", {})
            xl_config = model_config.get("xl", {})
            
            # Create xxs tier (1-5 images): Even more cautious than xs
            xxs_config = xs_config.copy()
            if xxs_config:
                xxs_config["max_train_epochs"] = xxs_config.get("max_train_epochs", 38) + 7  # 45 epochs
                xxs_config["train_batch_size"] = 2  # Smaller batch
                xxs_config["gradient_accumulation_steps"] = 4  # Higher accumulation
                if "unet_lr" in xxs_config:
                    xxs_config["unet_lr"] = xxs_config["unet_lr"] * 0.8  # Lower LR
                if "text_encoder_lr" in xxs_config:
                    xxs_config["text_encoder_lr"] = xxs_config["text_encoder_lr"] * 0.8
                xxs_config["noise_offset"] = 0.015  # Less noise
                xxs_config["lr_scheduler"] = "constant_with_warmup"
                xxs_config["max_data_loader_n_workers"] = 2
                if "weight_decay" in str(xxs_config.get("optimizer_args", [])):
                    # Increase regularization for small datasets
                    xxs_config["optimizer_args"] = [
                        "betas=(0.9, 0.999)",
                        "weight_decay=0.0002",
                        "eps=1e-08"
                    ]
                
                # Insert xxs at the beginning
                model_config = {"xxs": xxs_config, **model_config}
            
            # Create xxl tier (100+ images): Even more aggressive than xl
            xxl_config = xl_config.copy()
            if xxl_config:
                xxl_config["max_train_epochs"] = max(8, xxl_config.get("max_train_epochs", 11) - 3)  # 8 epochs
                xxl_config["train_batch_size"] = min(16, xxl_config.get("train_batch_size", 12) + 4)  # Larger batch
                xxl_config["gradient_accumulation_steps"] = 1
                if "unet_lr" in xxl_config:
                    xxl_config["unet_lr"] = min(5e-05, xxl_config["unet_lr"] * 1.15)  # Higher LR (capped)
                if "text_encoder_lr" in xxl_config:
                    xxl_config["text_encoder_lr"] = min(2.5e-06, xxl_config["text_encoder_lr"] * 1.15)
                xxl_config["noise_offset"] = 0.045  # More noise for diverse data
                xxl_config["lr_scheduler"] = "cosine"
                xxl_config["lr_warmup_steps"] = xxl_config.get("lr_warmup_steps", 120) + 30  # 150 steps
                xxl_config["max_data_loader_n_workers"] = 8
                
                # Add xxl at the end
                model_config["xxl"] = xxl_config
            
            # Update the model config with new tiers
            config_data["data"][model_hash] = model_config
    
    return config_data


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lrs_dir = os.path.join(script_dir, "lrs")
    
    # Update style_config.json
    style_config_path = os.path.join(lrs_dir, "style_config.json")
    print(f"Updating {style_config_path}...")
    with open(style_config_path, 'r') as f:
        style_config = json.load(f)
    
    style_config = add_enhanced_tiers(style_config)
    
    with open(style_config_path, 'w') as f:
        json.dump(style_config, f, indent=4)
    print(f"[OK] Updated style_config.json with xxs and xxl tiers")
    
    # Update person_config.json
    person_config_path = os.path.join(lrs_dir, "person_config.json")
    print(f"Updating {person_config_path}...")
    with open(person_config_path, 'r') as f:
        person_config = json.load(f)
    
    person_config = add_enhanced_tiers(person_config)
    
    with open(person_config_path, 'w') as f:
        json.dump(person_config, f, indent=4)
    print(f"[OK] Updated person_config.json with xxs and xxl tiers")
    
    print("\n[SUCCESS] Successfully added xxs and xxl tiers to all model configurations!")
    print("   xxs tier: 1-5 images (45 epochs, conservative)")
    print("   xxl tier: 100+ images (8 epochs, aggressive)")


if __name__ == "__main__":
    main()

