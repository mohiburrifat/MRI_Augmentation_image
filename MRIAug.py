#!/usr/bin/env python3
"""
MRI Image Augmentation Script (interactive) — fixed saving to avoid color/stack artifacts.

Run:
    python MRIAug.py
or with arguments:
    python MRIAug.py --input_dir /path/to/in --output_dir /path/to/out --n_augment 3

New features / fixes:
- Safe saving of PNGs: if the augmented result is a 3D volume (z,y,x) and we're saving PNG, the script
  saves the middle slice by default (to avoid PNG viewers interpreting the z axis as RGB channels).
- Option --save_slices to save every slice (creates many files).
"""
import os
import argparse
import numpy as np
from scipy import ndimage
from skimage import exposure
import nibabel as nib
import imageio
import warnings

warnings.filterwarnings('ignore')


def is_nifti(filename):
    return filename.lower().endswith('.nii') or filename.lower().endswith('.nii.gz')


def list_images(folder):
    exts = ('.nii', '.nii.gz', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
    except FileNotFoundError:
        return []
    return sorted(files)


def load_image(path):
    """
    Returns: data (np.ndarray float32), affine (or None), header (or None)
    For non-nifti images, returns 2D or 3D array depending on input (RGB images are HxWx3).
    """
    if is_nifti(path):
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)
        return data, img.affine, img.header
    else:
        arr = imageio.imread(path)
        data = np.asarray(arr, dtype=np.float32)
        # if RGB, keep channels (HxWx3). If grayscale, keep 2D.
        if data.ndim == 3 and data.shape[2] == 3:
            # keep as color image
            return data, None, None
        # otherwise ensure 2D grayscale
        if data.ndim == 3 and data.shape[2] in (4,):
            # RGBA -> drop alpha or convert to RGB
            data = data[..., :3].mean(axis=2) if data.shape[2] == 4 else data.mean(axis=2)
        if data.ndim == 3 and data.shape[2] != 3:
            # weird shape, collapse last axis by mean
            data = data.mean(axis=-1)
        return data, None, None


def _scale_to_uint8(img):
    """Scale a float image to uint8 [0,255] safely handling constant images and NaNs."""
    if not np.isfinite(img).any():
        return np.zeros(img.shape, dtype=np.uint8)
    valid = img[np.isfinite(img)]
    vmin, vmax = np.min(valid), np.max(valid)
    if vmax - vmin < 1e-8:
        # almost constant -> map to mid-gray
        return (np.ones_like(img) * 128).astype(np.uint8)
    out = (img - vmin) / (vmax - vmin)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def save_image(data, affine, header, src_path, out_path, save_slices=False):
    """
    Saves data respecting whether it's NIfTI (affine present) or a PNG/JPG.

    If `affine` is not None -> save as NIfTI (preserve header/affine).
    If affine is None and data is 2D -> save single PNG grayscale.
    If affine is None and data is 3D:
      - if last dim == 3 or 4 -> treat as RGB(A) image and save (clip to 0-255)
      - else treat as volume (z,y,x): by default save middle slice as PNG;
        if save_slices=True -> save all slices as name_slice{z:03d}.png
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if affine is not None:
        # Save as NIfTI preserving affine/header
        nif = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine, header=header)
        nib.save(nif, out_path)
        return

    # No affine -> save as PNG/JPG. Determine extension
    _, ext = os.path.splitext(out_path)
    ext = ext.lower()
    # If data is 2D -> single grayscale image
    if data.ndim == 2:
        arr8 = _scale_to_uint8(data)
        imageio.imwrite(out_path, arr8)
        return

    # If last axis is 3 or 4 -> interpret as color image HxWx3 or HxWx4
    if data.ndim == 3 and data.shape[2] in (3, 4):
        # If floats, scale each channel to 0-255 using global min/max
        arr = data
        # If channel values are already 0-255 and ints, just cast
        if np.issubdtype(arr.dtype, np.floating):
            # scale entire array using global finite min/max
            arr8 = _scale_to_uint8(arr)
            # _scale_to_uint8 collapsed channels; to preserve channels scale per-channel
            # Better: per-channel scaling if floats:
            chs = []
            for c in range(arr.shape[2]):
                chs.append(_scale_to_uint8(arr[..., c]))
            arr8 = np.stack(chs, axis=-1)
        else:
            arr8 = np.clip(arr, 0, 255).astype(np.uint8)
        imageio.imwrite(out_path, arr8)
        return

    # If 3D volume (z,y,x)
    if data.ndim == 3:
        zdim = data.shape[0]
        if save_slices:
            # save every slice
            base, _ = os.path.splitext(out_path)
            for zi in range(zdim):
                slice_img = data[zi]
                arr8 = _scale_to_uint8(slice_img)
                slice_path = f"{base}_slice{zi:03d}.png"
                imageio.imwrite(slice_path, arr8)
            return
        else:
            # save middle slice to avoid viewers interpreting z as channels
            mid = zdim // 2
            slice_img = data[mid]
            arr8 = _scale_to_uint8(slice_img)
            imageio.imwrite(out_path, arr8)
            return

    # Fallback (shouldn't normally reach)
    arr8 = _scale_to_uint8(np.asarray(data))
    imageio.imwrite(out_path, arr8)


# --- Augmentation primitives (conservative) ---


def apply_affine_transform(volume, rot_deg=0.0, shift=(0, 0, 0), scale=1.0):
    """
    volume: 2D (H,W) or 3D (Z,H,W). Rotation and in-plane zoom apply per-slice.
    shift should match the number of axes in the volume (for 3D: (z,y,x) or for 2D: (y,x)).
    """
    is_2d = (volume.ndim == 2)
    if is_2d:
        vol = volume[np.newaxis, :, :]
    else:
        vol = volume

    # scaling: apply in-plane only for 3D (i.e., don't scale along Z)
    if vol.ndim == 3:
        zoom_factors = (1.0, scale, scale)
    else:
        zoom_factors = (scale, scale)

    vol = ndimage.zoom(vol, zoom_factors, order=3, mode='nearest')

    if vol.ndim == 3:
        rotated = np.empty_like(vol)
        for i in range(vol.shape[0]):
            rotated[i] = ndimage.rotate(vol[i], rot_deg, reshape=False, order=3, mode='nearest')
    else:
        rotated = ndimage.rotate(vol, rot_deg, reshape=False, order=3, mode='nearest')

    # shift must be length equal to vol.ndim
    # If user provided shorter shift (legacy), pad with zeros on left (z axis)
    if isinstance(shift, (list, tuple)) and len(shift) != rotated.ndim:
        # assume shift is (y,x) for 2D, else (z,y,x) for 3D. Construct a proper tuple:
        if rotated.ndim == 3 and len(shift) == 2:
            shift_full = (0.0,) + tuple(shift)
        elif rotated.ndim == 2 and len(shift) == 3:
            # take last two
            shift_full = tuple(shift[-2:])
        else:
            # fallback zero
            shift_full = tuple([0.0] * rotated.ndim)
    else:
        shift_full = tuple(shift)

    shifted = ndimage.shift(rotated, shift=shift_full, order=3, mode='nearest')

    if is_2d:
        return shifted[0]
    return shifted


def add_gaussian_noise(img, sigma=0.01, rng=None):
    if rng is None:
        rng = np.random
    finite = np.isfinite(img)
    if not finite.any():
        return img
    scale = np.nanmax(img) - np.nanmin(img)
    if scale == 0:
        return img
    noise = rng.normal(0, sigma * scale, size=img.shape)
    return img + noise


def intensity_perturb(img, gamma_range=(0.9, 1.1), clip_percentile=0.5, rng=None):
    if rng is None:
        rng = np.random
    gamma = rng.uniform(gamma_range[0], gamma_range[1])
    mn, mx = np.nanmin(img), np.nanmax(img)
    if mx - mn == 0:
        return img
    norm = (img - mn) / (mx - mn)
    norm = np.power(norm, gamma)
    out = norm * (mx - mn) + mn
    p_low, p_high = np.percentile(out, [clip_percentile, 100 - clip_percentile])
    out = exposure.rescale_intensity(out, in_range=(p_low, p_high))
    return out


def elastic_deformation(volume, alpha=1.0, sigma=8.0, rng=None):
    if rng is None:
        rng = np.random
    if volume.ndim == 2:
        shape = volume.shape
        dx = rng.randn(*shape) * alpha
        dy = rng.randn(*shape) * alpha
        dx = ndimage.gaussian_filter(dx, sigma, mode='reflect')
        dy = ndimage.gaussian_filter(dy, sigma, mode='reflect')
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).ravel(), (x + dx).ravel()
        return ndimage.map_coordinates(volume, indices, order=3, mode='reflect').reshape(shape)
    else:
        shape = volume.shape
        dx = rng.randn(*shape) * alpha
        dy = rng.randn(*shape) * alpha
        dz = rng.randn(*shape) * alpha
        dx = ndimage.gaussian_filter(dx, sigma, mode='reflect')
        dy = ndimage.gaussian_filter(dy, sigma, mode='reflect')
        dz = ndimage.gaussian_filter(dz, sigma, mode='reflect')
        z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        zi = (z + dz).ravel()
        yi = (y + dy).ravel()
        xi = (x + dx).ravel()
        deformed = ndimage.map_coordinates(volume, [zi, yi, xi], order=3, mode='reflect')
        return deformed.reshape(shape)


def make_one_augmentation(data, rng=None):
    if rng is None:
        rng = np.random.RandomState()

    out = data.copy()

    rot = rng.uniform(-4.0, 4.0)  # degrees
    if out.ndim == 3:
        # assume (z,y,x)
        shift = (rng.uniform(-1.5, 1.5), rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5))
    else:
        shift = (rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5))
    scale = rng.uniform(0.985, 1.015)

    out = apply_affine_transform(out, rot_deg=rot, shift=shift, scale=scale)

    if rng.rand() < 0.3:
        out = elastic_deformation(out, alpha=0.5, sigma=12.0, rng=rng)

    out = intensity_perturb(out, gamma_range=(0.95, 1.05), clip_percentile=0.5, rng=rng)
    out = add_gaussian_noise(out, sigma=0.005, rng=rng)
    out = ndimage.gaussian_filter(out, sigma=0.3)

    return out


def augment_file(path, out_dir, n_augment=2, seed=None, save_slices=False):
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    if ext == '.gz':
        name = os.path.splitext(name)[0]
        ext = '.nii.gz'

    data, affine, header = load_image(path)
    rng = np.random.RandomState(seed)

    for i in range(n_augment):
        aug = make_one_augmentation(data, rng=rng)
        # choose out name extension: keep original extension if saving same type, else default to .png for non-nifti
        if affine is not None:
            out_name = f"{name}_aug{i+1}{ext}"
        else:
            # if original was a png/jpg, keep .png; if input was .nii but we ended up here, fallback to .nii
            out_name = f"{name}_aug{i+1}.png"
        out_path = os.path.join(out_dir, out_name)
        save_image(aug, affine, header, path, out_path, save_slices=save_slices)


def interactive_prompt_for_paths(default_input=None, default_output=None):
    while True:
        inp = input(f"Enter INPUT folder path [{default_input if default_input else ''}]: ").strip().strip('"')
        if inp == '' and default_input:
            inp = default_input
        if inp:
            if os.path.isdir(inp):
                break
            else:
                print("  Path not found or not a directory. Try again.")
        else:
            print("  Please enter a valid path.")
    while True:
        out = input(f"Enter OUTPUT folder path [{default_output if default_output else ''}]: ").strip().strip('"')
        if out == '' and default_output:
            out = default_output
        if out:
            try:
                os.makedirs(out, exist_ok=True)
                break
            except Exception as e:
                print("  Could not create output folder:", e)
        else:
            print("  Please enter a valid path.")
    return inp, out


def main():
    parser = argparse.ArgumentParser(description='Mild MRI image augmenter (interactive if args omitted)')
    parser.add_argument('--input_dir', '-i', required=False, help='Folder with input images (NIfTI or PNG/JPG)')
    parser.add_argument('--output_dir', '-o', required=False, help='Folder to save augmented images')
    parser.add_argument('--n_augment', '-n', type=int, default=None, help='Number of augmentations per image')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (optional)')
    parser.add_argument('--save_slices', action='store_true', help='If set, save every slice of 3D volumes as separate PNGs')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    n_augment = args.n_augment if args.n_augment is not None else 2
    save_slices_flag = args.save_slices

    if not input_dir or not output_dir:
        print("\n=== MRI Image Augmentation Tool ===")
        default_in = input_dir
        default_out = output_dir
        input_dir, output_dir = interactive_prompt_for_paths(default_input=default_in, default_output=default_out)
        if args.n_augment is None:
            try:
                n_augment_in = input("Enter number of augmentations per image [2]: ").strip()
                n_augment = int(n_augment_in) if n_augment_in != '' else 2
            except Exception:
                n_augment = 2

    if not os.path.isdir(input_dir):
        print("Input directory does not exist:", input_dir)
        return
    os.makedirs(output_dir, exist_ok=True)

    files = list_images(input_dir)
    if not files:
        print('No supported image files found in', input_dir)
        return

    print(f"\nFound {len(files)} image(s) in: {input_dir}")
    print(f"Saving {n_augment} augmented image(s) per file to: {output_dir}\n")

    for idx, f in enumerate(files, start=1):
        print(f'[{idx}/{len(files)}] Augmenting: {os.path.basename(f)}', end='', flush=True)
        try:
            augment_file(f, output_dir, n_augment=n_augment, seed=(None if args.seed is None else args.seed + idx),
                         save_slices=save_slices_flag)
            print("  ✓")
        except Exception as e:
            print(f"  ✗ failed -> {e}")

    print('\nDone. Augmented images saved to', output_dir)


if __name__ == '__main__':
    main()
