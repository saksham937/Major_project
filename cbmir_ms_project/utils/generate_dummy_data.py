import os
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

def create_dummy_nifti(filepath, shape=(128, 128, 30), num_lesions=3, has_lesion=True):
    """
    Creates a dummy MRI volume and a corresponding lesion mask.
    Returns: volume data, mask data
    """
    # Create background brain-like shape (ellipsoid)
    z, y, x = np.ogrid[-shape[0]//2:shape[0]//2, -shape[1]//2:shape[1]//2, -shape[2]//2:shape[2]//2]
    mask_brain = (x**2 + y**2 + z**2) <= (min(shape)//2.2)**2
    
    # Base intensity
    volume = np.zeros(shape, dtype=np.float32)
    volume[mask_brain] = 0.5 + 0.1 * np.random.randn(*volume[mask_brain].shape)
    
    # Add tissue structure
    volume = gaussian_filter(volume, sigma=2)
    
    mask = np.zeros(shape, dtype=np.uint8)
    
    if has_lesion:
        for _ in range(num_lesions):
            # Random lesion center within the brain
            cx = np.random.randint(shape[0]//4, 3*shape[0]//4)
            cy = np.random.randint(shape[1]//4, 3*shape[1]//4)
            cz = np.random.randint(shape[2]//4, 3*shape[2]//4)
            
            # Lesion size
            r = np.random.randint(2, 6)
            
            lz, ly, lx = np.ogrid[-cx:shape[0]-cx, -cy:shape[1]-cy, -cz:shape[2]-cz]
            lesion_mask = (lx**2 + ly**2 + lz**2) <= r**2
            
            # Bright lesions (like in FLAIR or T2)
            volume[lesion_mask] += 0.4 
            mask[lesion_mask] = 1

    # Normalize volume
    volume = np.clip(volume, 0, 1)

    # Save as NIfTI
    vol_nii = nib.Nifti1Image(volume, np.eye(4))
    mask_nii = nib.Nifti1Image(mask, np.eye(4))

    nib.save(vol_nii, filepath)
    nib.save(mask_nii, filepath.replace('.nii.gz', '_mask.nii.gz'))
    
    return volume, mask

def generate_dataset(base_dir, num_patients=10):
    print(f"Generating dummy ISBI dataset at: {base_dir}")
    os.makedirs(base_dir, exist_ok=True)
    
    for i in range(1, num_patients + 1):
        patient_id = f"training{i:02d}"
        patient_dir = os.path.join(base_dir, patient_id)
        os.makedirs(patient_dir, exist_ok=True)
        
        # 80% chance for a patient to have lesions
        has_lesion = np.random.rand() > 0.2
        num_lesions = np.random.randint(1, 5) if has_lesion else 0
        
        # Generate modalities
        for modality in ['flair', 't1', 't2']:
            filepath = os.path.join(patient_dir, f"{modality}.nii.gz")
            create_dummy_nifti(filepath, num_lesions=num_lesions, has_lesion=has_lesion)
            
        print(f"Generated {patient_id} - Lesions: {has_lesion} ({num_lesions})")

if __name__ == "__main__":
    # Get current script dir, then point to data
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'dummy_isbi'))
    generate_dataset(base_dir, num_patients=15)
    print("Done! Dummy dataset ready for testing.")
