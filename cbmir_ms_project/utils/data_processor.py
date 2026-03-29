import os
import glob
import numpy as np
import nibabel as nib
from PIL import Image
import torch
from torchvision import transforms

class MedicalImageProcessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
        # Standard transforms for ResNet input
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_nifti(self, filepath):
        """Loads a NIfTI file and returns its numpy array."""
        try:
            img = nib.load(filepath)
            data = img.get_fdata()
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def normalize_intensity(self, volume):
        """Normalizes volume intensities to 0-255 range."""
        min_val = np.min(volume)
        max_val = np.max(volume)
        
        if max_val - min_val == 0:
            return np.zeros_like(volume, dtype=np.uint8)
            
        normalized = (volume - min_val) / (max_val - min_val) * 255.0
        return normalized.astype(np.uint8)

    def extract_axial_slices(self, volume, num_slices=None):
        """
        Extracts 2D axial slices from a 3D MRI volume.
        If num_slices is provided, extracts that many uniformly spaced slices from the center region.
        """
        # Axial slices are typically along the Z axis (last dimension in common NIfTI)
        z_dim = volume.shape[2]
        
        if num_slices is None:
            # Extract all slices
            slices = [volume[:, :, i] for i in range(z_dim)]
            indices = list(range(z_dim))
        else:
            # Avoid the very edges which are often empty
            start = int(z_dim * 0.2)
            end = int(z_dim * 0.8)
            if end <= start:
                start, end = 0, z_dim
            indices = np.linspace(start, end - 1, num_slices, dtype=int)
            slices = [volume[:, :, i] for i in indices]
            
        return slices, indices

    def preprocess_slice_for_model(self, slice_2d):
        """
        Converts a 2D slice (grayscale) into a 3-channel tensor suitable for CNNs.
        """
        # Convert to 3 channels (RGB) by duplicating the grayscale channel
        if len(slice_2d.shape) == 2:
            slice_rgb = np.stack((slice_2d,)*3, axis=-1)
        else:
            slice_rgb = slice_2d
            
        # Apply standard ImageNet transforms
        tensor = self.transform(slice_rgb)
        
        # Add batch dimension
        return tensor.unsqueeze(0)

def extract_dataset_slices(data_dir, output_dir, modalities=['flair', 't2']):
    """
    Utility func to parse the ISBI dataset structure and extract 2D JPEGs for indexing.
    Supports both the dummy dataset layout and the real ISBI layout.
    """
    processor = MedicalImageProcessor()
    os.makedirs(output_dir, exist_ok=True)
    
    patient_dirs = glob.glob(os.path.join(data_dir, "training*"))
    slice_metadata = []
    
    print(f"Found {len(patient_dirs)} patients.")
    for p_dir in patient_dirs:
        p_id = os.path.basename(p_dir)
        
        # Check if real ISBI layout exists: preprocessed/ and masks/
        is_real_isbi = os.path.exists(os.path.join(p_dir, "preprocessed"))
        
        if is_real_isbi:
            # We look in preprocessed folder
            prep_dir = os.path.join(p_dir, "preprocessed")
            masks_dir = os.path.join(p_dir, "masks")
            
            all_files = glob.glob(os.path.join(prep_dir, "*_pp.nii"))
            
            for fpath in all_files:
                fname = os.path.basename(fpath) # e.g. training01_01_flair_pp.nii
                parts = fname.split('_')
                if len(parts) >= 3:
                    timepoint = parts[1]
                    mod = parts[2]
                    
                    if mod not in modalities:
                        continue
                        
                    vol_path = fpath
                    # Look for mask1.nii
                    mask_path = os.path.join(masks_dir, f"{p_id}_{timepoint}_mask1.nii")
                    
                    vol = processor.load_nifti(vol_path)
                    vol_norm = processor.normalize_intensity(vol)
                    
                    mask = None
                    if os.path.exists(mask_path):
                        mask = processor.load_nifti(mask_path)
                    
                    # Extract 15 middle slices
                    slices, indices = processor.extract_axial_slices(vol_norm, num_slices=15)
                    
                    for i, (slc, idx) in enumerate(zip(slices, indices)):
                        slice_filename = f"{p_id}_{timepoint}_{mod}_slice{idx:03d}.jpg"
                        slice_path = os.path.join(output_dir, slice_filename)
                        
                        has_lesion = False
                        if mask is not None:
                            mask_slice = mask[:, :, idx]
                            if np.sum(mask_slice) > 0:
                                has_lesion = True
                                
                        im = Image.fromarray(slc)
                        im.save(slice_path)
                        
                        slice_metadata.append({
                            'id': slice_filename,
                            'patient_id': f"{p_id}_{timepoint}",
                            'modality': mod,
                            'slice_index': idx,
                            'path': slice_path,
                            'has_lesion': has_lesion
                        })
        else:
            # Dummy dataset layout logic fallback
            for mod in modalities:
                vol_path = os.path.join(p_dir, f"{mod}.nii.gz")
                mask_path = os.path.join(p_dir, f"{mod}_mask.nii.gz")
                
                if not os.path.exists(vol_path):
                    continue
                    
                vol = processor.load_nifti(vol_path)
                vol_norm = processor.normalize_intensity(vol)
                
                mask = None
                if os.path.exists(mask_path):
                    mask = processor.load_nifti(mask_path)
                
                slices, indices = processor.extract_axial_slices(vol_norm, num_slices=15)
                
                for i, (slc, idx) in enumerate(zip(slices, indices)):
                    slice_filename = f"{p_id}_{mod}_slice{idx:03d}.jpg"
                    slice_path = os.path.join(output_dir, slice_filename)
                    
                    has_lesion = False
                    if mask is not None:
                        mask_slice = mask[:, :, idx]
                        if np.sum(mask_slice) > 0:
                            has_lesion = True
                            
                    im = Image.fromarray(slc)
                    im.save(slice_path)
                    
                    slice_metadata.append({
                        'id': slice_filename,
                        'patient_id': p_id,
                        'modality': mod,
                        'slice_index': idx,
                        'path': slice_path,
                        'has_lesion': has_lesion
                    })
                    
    return slice_metadata

if __name__ == "__main__":
    # Test
    processor = MedicalImageProcessor()
    # If dummy data exists
    test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'dummy_isbi', 'training01', 'flair.nii.gz'))
    if os.path.exists(test_file):
        vol = processor.load_nifti(test_file)
        print(f"Loaded volume shape: {vol.shape}")
        norm = processor.normalize_intensity(vol)
        slices, idx = processor.extract_axial_slices(norm, 5)
        print(f"Extracted {len(slices)} slices.")
        tensor = processor.preprocess_slice_for_model(slices[0])
        print(f"Processed tensor shape: {tensor.shape}")
