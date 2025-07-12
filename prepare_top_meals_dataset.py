"""
Prepare Top Meals Dataset for LoRA Training

This script:
1. Reads the top_k_meals_summary.csv 
2. Selects the top N dishes (default: 2)
3. Loads images and metadata from each dish's CSV file
4. Creates a combined dataset CSV for training
5. Validates that images exist and reports statistics
"""

import os
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Tuple
import shutil

def validate_image_paths(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Validate that image paths exist and return statistics"""
    stats = {
        'total_entries': len(df),
        'existing_images': 0,
        'missing_images': 0,
        'unique_images': 0
    }
    
    valid_rows = []
    existing_images = set()
    
    for idx, row in df.iterrows():
        image_path = row['image_path']
        if os.path.exists(image_path):
            stats['existing_images'] += 1
            existing_images.add(image_path)
            valid_rows.append(row)
        else:
            stats['missing_images'] += 1
            print(f"❌ Missing image: {image_path}")
    
    stats['unique_images'] = len(existing_images)
    valid_df = pd.DataFrame(valid_rows)
    
    return valid_df, stats

def create_training_prompt(row: pd.Series) -> str:
    """Create a detailed training prompt from meal data"""
    parts = [
        str(row.get("description", "")).strip(),
        "with side salad" if str(row.get("Beilagensalat", "")).lower() in ["yes", "ja"] else "",
        "with regional apple" if str(row.get("Regio Apfel", "")).lower() in ["yes", "ja"] else "",
        f"meal type: {row.get('type', '')}",
        f"diet: {row.get('diet', '')}",
        f"served at {row.get('mensa', '')}"
    ]
    
    prompt = ", ".join([p for p in parts if p])
    return prompt

def prepare_dataset(data_root: str, top_n: int = 2, max_images_per_dish: int = None):
    """Prepare dataset from top N dishes"""
    
    # Read the summary file
    summary_path = os.path.join(data_root, "top_k_meals_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    summary_df = pd.read_csv(summary_path)
    top_dishes = summary_df.head(top_n)
    
    print(f"🍽️  Preparing dataset for top {top_n} dishes:")
    for idx, dish in top_dishes.iterrows():
        print(f"   {idx+1}. {dish['description']} ({dish['occurrences']} occurrences)")
    
    # Collect all meal data
    all_meals = []
    total_images = 0
    
    for idx, dish in top_dishes.iterrows():
        slice_path = os.path.join(data_root, dish['slice_path'])
        
        if not os.path.exists(slice_path):
            print(f"⚠️  Slice file not found: {slice_path}")
            continue
            
        # Read the dish's CSV file
        dish_df = pd.read_csv(slice_path)
        
        # Limit images per dish if specified
        if max_images_per_dish:
            dish_df = dish_df.head(max_images_per_dish)
        
        print(f"📊 Loaded {len(dish_df)} entries for: {dish['description']}")
        all_meals.extend(dish_df.to_dict('records'))
        total_images += len(dish_df)
    
    # Create combined dataframe
    combined_df = pd.DataFrame(all_meals)
    print(f"📈 Total entries before validation: {len(combined_df)}")
    
    # Validate image paths
    valid_df, stats = validate_image_paths(combined_df)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total entries: {stats['total_entries']}")
    print(f"   Existing images: {stats['existing_images']}")
    print(f"   Missing images: {stats['missing_images']}")
    print(f"   Unique images: {stats['unique_images']}")
    print(f"   Success rate: {stats['existing_images']/stats['total_entries']*100:.1f}%")
    
    if len(valid_df) == 0:
        raise ValueError("No valid images found!")
    
    # Add training prompts
    valid_df['training_prompt'] = valid_df.apply(create_training_prompt, axis=1)
    
    # Show sample prompts
    print(f"\n📝 Sample prompts:")
    for i, row in valid_df.head(3).iterrows():
        print(f"   {i+1}. {row['training_prompt'][:100]}...")
    
    return valid_df, stats

def copy_images_to_dataset_folder(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Copy images to a local dataset folder and update paths"""
    dataset_images_dir = os.path.join(output_dir, "images")
    os.makedirs(dataset_images_dir, exist_ok=True)
    
    updated_rows = []
    copied_count = 0
    
    for idx, row in df.iterrows():
        src_path = row['image_path']
        
        if os.path.exists(src_path):
            # Create a unique filename
            filename = f"{row['mensa'].replace(' ', '_')}_{row['type'].replace(' ', '_')}_{Path(src_path).stem}_{idx}.jpg"
            filename = filename.replace('/', '_').replace('\\', '_')  # Clean filename
            
            dst_path = os.path.join(dataset_images_dir, filename)
            
            try:
                shutil.copy2(src_path, dst_path)
                row_copy = row.copy()
                row_copy['image_path'] = dst_path  # Update to local path
                updated_rows.append(row_copy)
                copied_count += 1
            except Exception as e:
                print(f"❌ Failed to copy {src_path}: {e}")
    
    print(f"📁 Copied {copied_count} images to {dataset_images_dir}")
    return pd.DataFrame(updated_rows)

def main():
    parser = argparse.ArgumentParser(description='Prepare top meals dataset for LoRA training')
    parser.add_argument('--data_root', default='/work/dlclarge2/ceylanb-DL_Lab_Project/data', 
                       help='Path to data directory')
    parser.add_argument('--output_dir', default='/work/dlclarge2/ceylanb-DL_Lab_Project/mensa-lora/dataset', 
                       help='Output directory for prepared dataset')
    parser.add_argument('--top_n', type=int, default=2, 
                       help='Number of top dishes to include (default: 2)')
    parser.add_argument('--max_images_per_dish', type=int, default=20,
                       help='Maximum images per dish (default: 20, None for all)')
    parser.add_argument('--copy_images', action='store_true',
                       help='Copy images to local dataset folder')
    
    args = parser.parse_args()
    
    print(f"🚀 Preparing dataset with top {args.top_n} dishes")
    print(f"📂 Data root: {args.data_root}")
    print(f"📁 Output dir: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare dataset
    df, stats = prepare_dataset(
        data_root=args.data_root,
        top_n=args.top_n,
        max_images_per_dish=args.max_images_per_dish
    )
    
    # Optionally copy images locally
    if args.copy_images:
        print("\n📁 Copying images to local dataset folder...")
        df = copy_images_to_dataset_folder(df, args.output_dir)
    
    # Save the dataset CSV
    output_csv = os.path.join(args.output_dir, "dataset.csv")
    
    # Select relevant columns for training
    training_df = df[['training_prompt', 'image_path', 'description', 'type', 'diet', 'mensa']].copy()
    training_df.rename(columns={'training_prompt': 'text'}, inplace=True)
    
    training_df.to_csv(output_csv, index=False)
    print(f"\n✅ Dataset saved to: {output_csv}")
    print(f"📊 Final dataset size: {len(training_df)} samples")
    
    # Show memory estimation for RTX 2080 (12GB)
    estimated_memory_per_image = 150  # MB rough estimate for 512x512 training
    total_memory_estimate = len(training_df) * estimated_memory_per_image / 1024  # GB
    print(f"\n💾 Estimated VRAM usage: {total_memory_estimate:.1f} GB (excluding model weights)")
    
    if total_memory_estimate > 8:  # Leave some headroom
        print("⚠️  High memory usage detected. Consider reducing dataset size or batch size.")
    else:
        print("✅ Memory usage should fit within RTX 2080 constraints.")

if __name__ == "__main__":
    main()
