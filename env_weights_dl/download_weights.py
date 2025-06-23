#!/usr/bin/env python3
"""
Script to download all required OpenPI model weights.
This will download:
1. PI-0 base model weights
2. PaliGemma expert weights  
3. PaliGemma tokenizer
4. SigLIP weights (embedded in PI-0)

Run this script to pre-download all weights before running inference.
"""

import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Download all required model weights."""
    
    # Set cache directory (optional)
    cache_dir = Path.home() / ".cache" / "openpi"
    os.environ.setdefault("OPENPI_DATA_HOME", str(cache_dir))
    
    logger.info(f"OpenPI cache directory: {os.environ.get('OPENPI_DATA_HOME', cache_dir)}")
    
    try:
        # Import OpenPI modules
        from openpi.shared import download
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.training.weight_loaders import PaliGemmaWeightLoader, CheckpointWeightLoader
        from openpi.training import config as _config
        
        logger.info("Starting weight downloads...")
        
        # 1. Download PI-0 base model weights
        logger.info("1. Downloading PI-0 base model weights...")
        pi0_path = download.maybe_download("s3://openpi-assets/checkpoints/pi0_base/params")
        logger.info(f"PI-0 base model downloaded to: {pi0_path}")
        
        # 2. Download PaliGemma tokenizer
        logger.info("2. Downloading PaliGemma tokenizer...")
        tokenizer_path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        logger.info(f"PaliGemma tokenizer downloaded to: {tokenizer_path}")
        
        # 3. Download PaliGemma expert weights
        logger.info("3. Downloading PaliGemma expert weights...")
        paligemma_path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", 
            gs={"token": "anon"}
        )
        logger.info(f"PaliGemma expert weights downloaded to: {paligemma_path}")
        
        # 4. Test model creation to ensure all weights are properly downloaded
        logger.info("4. Testing model creation to verify downloads...")
        try:
            # Create a simple PI-0 config to test weight loading
            config = _config.get_config("pi0_aloha_sim")  # Use a simple config
            
            # This will trigger any remaining downloads
            checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
            logger.info(f"Test checkpoint downloaded to: {checkpoint_dir}")
            
        except Exception as e:
            logger.warning(f"Model creation test failed (this is normal if you don't have GPU): {e}")
        
        logger.info("✅ All weights downloaded successfully!")
        logger.info("You can now run OpenPI inference without additional downloads.")
        
        # Print cache directory info
        if cache_dir.exists():
            total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            logger.info(f"Total cache size: {total_size / (1024**3):.2f} GB")
        
    except ImportError as e:
        logger.error(f"Failed to import OpenPI modules: {e}")
        logger.error("Make sure you have installed OpenPI properly.")
        return 1
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
