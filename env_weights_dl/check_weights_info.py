#!/usr/bin/env python3
"""
Script to check OpenPI model weights information and download status.
"""

import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def check_cache_status():
    """Check what's already downloaded in the cache."""
    cache_dir = Path.home() / ".cache" / "openpi"
    
    logger.info(f"Checking cache directory: {cache_dir}")
    
    if not cache_dir.exists():
        logger.info("❌ Cache directory doesn't exist yet")
        return
    
    # Check for different weight files
    weight_files = {
        "PI-0 Base Model": cache_dir / "openpi-assets" / "checkpoints" / "pi0_base",
        "PI-0 ALOHA Sim": cache_dir / "openpi-assets" / "checkpoints" / "pi0_aloha_sim",
        "PI-0 FAST Base": cache_dir / "openpi-assets" / "checkpoints" / "pi0_fast_base",
        "PI-0 FAST DROID": cache_dir / "openpi-assets" / "checkpoints" / "pi0_fast_droid",
        "PaliGemma Tokenizer": cache_dir / "big_vision" / "paligemma_tokenizer.model",
        "PaliGemma Weights": cache_dir / "vertex-model-garden-paligemma-us" / "paligemma" / "pt_224.npz",
    }
    
    total_size = 0
    for name, path in weight_files.items():
        if path.exists():
            if path.is_dir():
                size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                file_count = len([f for f in path.rglob('*') if f.is_file()])
                logger.info(f"✅ {name}: {format_size(size)} ({file_count} files)")
            else:
                size = path.stat().st_size
                logger.info(f"✅ {name}: {format_size(size)}")
            total_size += size
        else:
            logger.info(f"❌ {name}: Not downloaded")
    
    if total_size > 0:
        logger.info(f"📊 Total downloaded: {format_size(total_size)}")
    else:
        logger.info("📊 No weights downloaded yet")

def show_download_urls():
    """Show the URLs that will be downloaded."""
    logger.info("\n🔗 Download URLs:")
    
    urls = {
        "PI-0 Base Model": "s3://openpi-assets/checkpoints/pi0_base/params",
        "PI-0 FAST Base": "s3://openpi-assets/checkpoints/pi0_fast_base/params", 
        "PI-0 ALOHA Sim": "s3://openpi-assets/checkpoints/pi0_aloha_sim/params",
        "PI-0 FAST DROID": "s3://openpi-assets/checkpoints/pi0_fast_droid/params",
        "PaliGemma Tokenizer": "gs://big_vision/paligemma_tokenizer.model",
        "PaliGemma Expert": "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz",
    }
    
    for name, url in urls.items():
        logger.info(f"  {name}: {url}")

def main():
    """Main function."""
    logger.info("🤖 OpenPI Weights Information")
    logger.info("=" * 50)
    
    # Show cache directory
    cache_dir = Path.home() / ".cache" / "openpi"
    logger.info(f"Cache directory: {cache_dir}")
    
    # Check current status
    check_cache_status()
    
    # Show download URLs
    show_download_urls()
    
    logger.info("\n💡 Tips:")
    logger.info("1. Run 'python download_weights.py' to download all weights")
    logger.info("2. Set OPENPI_DATA_HOME environment variable to change cache location")
    logger.info("3. Weights are downloaded automatically when running inference")
    logger.info("4. For continuous download, ensure stable internet connection")

if __name__ == "__main__":
    main()
