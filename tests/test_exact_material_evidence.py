import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest


def module():
    path = Path(__file__).resolve().parents[1]/'scripts/label-material-training.py'
    spec = importlib.util.spec_from_file_location('material_teacher_exact', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def dataset(tmp_path, suffix='.jpg'):
    image = Image.new('RGB', (64, 64), 'red')
    crop = tmp_path/('crop'+suffix)
    image.save(crop)
    scene = tmp_path/'scene.jpg'
    image.save(scene)
    checksum = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'dataset.json').write_text(json.dumps({'entries': [{
        'id': 'one', 'file': crop.name, 'sha256': checksum(crop),
        'scene_file': scene.name, 'scene_sha256': checksum(scene),
        'bounds_xyxy': [0, 0, 64, 64]}]}))
    return crop


def test_exact_jpeg_evidence_preserves_source_bytes(tmp_path):
    crop = dataset(tmp_path)
    _, manifest = module().prepare(tmp_path, preserve_crop_jpeg=True)
    evidence = manifest['batches'][0]['images'][1]
    assert (tmp_path/evidence['file']).read_bytes() == crop.read_bytes()
    assert manifest['crop_encoding'] == 'exact-source-jpeg'
    with pytest.raises(ValueError, match='changed'):
        module().prepare(tmp_path, preserve_crop_jpeg=False)


def test_exact_jpeg_flag_rejects_format_substitution(tmp_path):
    dataset(tmp_path, '.png')
    with pytest.raises(ValueError, match='requires RGB JPEG'):
        module().prepare(tmp_path, preserve_crop_jpeg=True)


def test_paired_variants_never_share_teacher_request():
    rows=[{'id': f'{group}-{variant}', 'group_id': group}
          for group in ['a', 'b', 'c', 'd', 'e'] for variant in range(4)]
    batches=list(module().sample_batches(rows, unique_groups=True))
    assert sorted(row['id'] for batch in batches for row in batch) == sorted(row['id'] for row in rows)
    assert all(len({row['group_id'] for row in batch}) == len(batch) <= 4 for batch in batches)
    assert list(module().sample_batches(rows)) == [rows[i:i+4] for i in range(0,len(rows),4)]
