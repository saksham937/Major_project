import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_processor import extract_dataset_slices, MedicalImageProcessor
from models.feature_extractor import FeatureExtractor
from features.indexer import CBMIRIndexer

def build_index(data_dir, output_dir, model_name='resnet50', metric='cosine'):
    print("--- CBMIR Index Building Pipeline ---")
    
    # 1. Extract 2D slices from 3D NIfTI volumes
    slices_dir = os.path.join(output_dir, 'extracted_slices')
    print(f"\n1. Extracting slices to {slices_dir}...")
    metadata_list = extract_dataset_slices(data_dir, slices_dir, modalities=['flair', 't2'])
    
    if not metadata_list:
        print("No image slices extracted. Have you generated the dummy dataset?")
        return
        
    # 2. Extract Features
    print(f"\n2. Extracting features using {model_name}...")
    extractor = FeatureExtractor(model_name=model_name)
    processor = MedicalImageProcessor()
    
    all_features = []
    
    for meta in tqdm(metadata_list, desc="Processing images"):
        img_path = meta['path']
        try:
            # We load the JPG images we just extracted
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
            tensor = processor.transform(img_np).unsqueeze(0)
            
            # Extract
            feat = extractor(tensor)
            all_features.append(feat.cpu().numpy()[0])
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    all_features = np.array(all_features)
    
    # 3. Build Vector Index
    print(f"\n3. Building Vector Index ({metric})...")
    indexer = CBMIRIndexer(metric=metric)
    indexer.add_items(all_features, metadata_list)
    
    # 4. Save Index
    index_dir = os.path.join(output_dir, 'index')
    print(f"\n4. Saving index to {index_dir}...")
    indexer.save(index_dir, prefix=f"ms_cbmir_{model_name}_{metric}")
    
    print("\n--- Pipeline Complete! ---")
    print(f"Indexed {len(metadata_list)} images successfully.")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    real_data_dir = os.path.abspath(os.path.join(base_dir, '..')) # user's training dir
    output_dir = os.path.join(base_dir, 'data', 'processed')
    
    data_dir = real_data_dir
    # Check if the real ISBI dataset is there
    if not os.path.exists(os.path.join(real_data_dir, 'training_01', 'preprocessed')) and not os.path.exists(os.path.join(real_data_dir, 'training01', 'preprocessed')):
        print("Real ISBI dataset not found. Falling back to dummy data.")
        data_dir = os.path.join(base_dir, 'data', 'dummy_isbi')
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} not found. Running dummy data generator...")
            from utils.generate_dummy_data import generate_dataset
            generate_dataset(data_dir, num_patients=10)
    build_index(data_dir, output_dir, model_name='inception_v3', metric='chebyshev')
