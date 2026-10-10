"""Prepare and verify three fixed grayscale reference inputs.

Default mode verifies bundled metadata-free pixel arrays. --fetch downloads
the named scikit-image 0.25.2 source bytes, checks their SHA-256 digests, and
recreates the center crops. --source-dir accepts already retrieved PNGs.
No image interpolation, intensity transformation or random sampling occurs.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parent
URL_ROOT = 'https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/'
DOC_ROOT = 'https://scikit-image.org/docs/0.25.x/api/skimage.data.html'
SOURCES = {
    'camera': {
        'source_filename': 'camera.png',
        'source_shape': [512, 512],
        'source_bytes': 139512,
        'source_sha256': 'b0793d2adda0fa6ae899c03989482bff9a42d3d5690fc7e3648f2795d730c23a',
        'crop_origin_zero_based': [128, 128],
        'crop_pixel_sha256': '685445e0c73e742f8c7b9262e59192536d26cfecceabd3c3502539bfb5732626',
        'license': 'CC0-1.0',
        'credit': 'Lav Varshney',
        'description': 'Camera photograph; the CC0 replacement used since scikit-image 0.18.',
    },
    'clock': {
        'source_filename': 'clock_motion.png',
        'source_shape': [300, 400],
        'source_bytes': 58784,
        'source_sha256': 'f029226b28b642e80113d86622e9b215ee067a0966feaf5e60604a1e05733955',
        'crop_origin_zero_based': [22, 72],
        'crop_pixel_sha256': 'a68a3c89ff40d7dea57c6915ef866fd493088eabe63e5cb8f5c5567772815718',
        'license': 'Public domain dedication by the photographer, as documented by scikit-image.',
        'credit': 'Stefan van der Walt',
        'description': 'Motion-blurred wall clock photograph.',
    },
    'grass': {
        'source_filename': 'grass.png',
        'source_shape': [512, 512],
        'source_bytes': 217893,
        'source_sha256': 'b6b6022426b38936c43a4ac09635cd78af074e90f42ffa8227ac8b7452d39f89',
        'crop_origin_zero_based': [128, 128],
        'crop_pixel_sha256': '8f48099c41b85fa495c03a7c868f6ff7a176940d96ccb90d7b0c6594456da7cc',
        'license': 'CC0-1.0',
        'credit': 'linolafett, Grass 01, DeviantArt',
        'original_source_url': 'https://www.deviantart.com/linolafett/art/Grass-01-434853879',
        'description': 'Grayscale grass texture distributed by scikit-image.',
    },
}
PERMUTATION_NAMESPACE = b'fixed-image-permutation-v1\x00'


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def permutation_indices(size=65536):
    """Version-independent deterministic permutation, shared by all inputs.

    Sort original row-major indices by SHA-256(namespace || uint32be(index)),
    breaking an unlikely digest tie by index. This constructs one fixed
    spatial permutation; it is not an additional stochastic experimental unit.
    """
    return np.array(sorted(range(size), key=lambda i: (
        hashlib.sha256(PERMUTATION_NAMESPACE+i.to_bytes(4, 'big')).digest(), i)), dtype=np.int64)


def load_inputs(root=ROOT):
    inputs = {}
    for name, item in SOURCES.items():
        a = np.load(Path(root)/'inputs'/f'{name}.npy', allow_pickle=False)
        assert a.dtype == np.uint8 and a.shape == (256, 256), name
        assert sha256(a.tobytes(order='C')) == item['crop_pixel_sha256'], name
        inputs[name] = a
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--fetch', action='store_true')
    mode.add_argument('--source-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    target = args.output_dir
    if args.fetch or args.source_dir:
        from PIL import Image
        (target/'inputs').mkdir(parents=True, exist_ok=True)
        for name, item in SOURCES.items():
            if args.fetch:
                with urllib.request.urlopen(URL_ROOT+item['source_filename'], timeout=60) as response:
                    raw = response.read()
            else:
                raw = (args.source_dir/item['source_filename']).read_bytes()
            assert len(raw) == item['source_bytes'], name
            assert sha256(raw) == item['source_sha256'], name
            with Image.open(io.BytesIO(raw)) as image:
                assert image.mode == 'L', name
                a = np.asarray(image)
                assert list(a.shape) == item['source_shape'] and a.dtype == np.uint8, name
                r, c = item['crop_origin_zero_based']
                crop = a[r:r+256, c:c+256].copy()
            assert sha256(crop.tobytes(order='C')) == item['crop_pixel_sha256'], name
            np.save(target/'inputs'/f'{name}.npy', crop, allow_pickle=False)
    inputs = load_inputs(target)
    permutation = permutation_indices()
    assert np.array_equal(np.sort(permutation), np.arange(65536))
    records = {}
    for name, a in inputs.items():
        b = a.ravel()[permutation].reshape(a.shape)
        assert np.array_equal(np.bincount(a.ravel(), minlength=256), np.bincount(b.ravel(), minlength=256))
        item = dict(SOURCES[name])
        item.update({
            'source_url': URL_ROOT+item['source_filename'],
            'license_documentation_url': DOC_ROOT+'#skimage.data.'+name,
            'crop_shape': [256, 256],
            'crop_file_sha256': sha256((target/'inputs'/f'{name}.npy').read_bytes()),
            'permuted_pixel_sha256': sha256(b.tobytes(order='C')),
            'histogram_preserved': True,
        })
        records[name] = item
    manifest = {
        'upstream_version': 'scikit-image 0.25.2',
        'retrieval_date_utc': '2026-10-06',
        'preprocessing': 'Centered 256x256 crop in the supplied 8-bit grayscale image; no interpolation or intensity transformation.',
        'statistical_role': 'Fixed reference images compared with independent uniform 8-bit comparison pixels.',
        'permutation': 'Sort row-major indices by SHA-256 of ASCII fixed-image-permutation-v1, a zero byte, and a 4-byte big-endian index; tie-break by index. Apply the same permutation to every crop.',
        'images': records,
    }
    (target/'input_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Verified three 256x256 uint8 inputs, source crop digests and histogram-preserving permutation.')


if __name__ == '__main__':
    main()
