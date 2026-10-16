import json
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file
import torch

from flagos_compressor.cli.repack import repack_w8a16
from flagos_compressor.core.validation import validate_artifact
from flagos_compressor.formats.int8_pack import unpack_uint8b128_int32


def source_checkpoint(path: Path):
    path.mkdir()
    weights = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).reshape(8, 32)
    scales = torch.linspace(0.01, 0.08, 8, dtype=torch.float32).reshape(8, 1)
    router = torch.tensor([[1.23456789, -0.123456789]], dtype=torch.float32)
    save_file({'model.q_proj.weight': weights, 'model.q_proj.weight_scale': scales,
               'model.gate.weight': router, 'model.scalar': torch.tensor(1.23456789)},
              path / 'model-00001-of-00001.safetensors')
    (path / 'config.json').write_text(json.dumps({'quantization_config': {
        'quant_method': 'compressed-tensors', 'format': 'int-quantized',
        'config_groups': {'int8': {'targets': ['model.q_proj'], 'weights': {
            'num_bits': 8, 'type': 'int', 'strategy': 'channel', 'symmetric': True,
            'dynamic': False}, 'input_activations': {'num_bits': 8, 'type': 'int',
            'dynamic': True, 'strategy': 'token'}}}}}))
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
        name: 'model-00001-of-00001.safetensors' for name in
        ['model.q_proj.weight', 'model.q_proj.weight_scale', 'model.gate.weight', 'model.scalar']}}))
    (path / 'quantization_manifest.json').write_text(json.dumps({'schema': 'flagos-compressor.provenance.v1', 'tensors': {
        'model.q_proj.weight': {'format': 'compressed-tensors-int-quantized-int8',
            'logical_name': 'model.q_proj.weight', 'num_bits': 8, 'strategy': 'channel',
            'scale': 'model.q_proj.weight_scale', 'scale_shape': [8, 1],
            'scale_dtype': 'float32', 'logical_shape': [8, 32]}}}))
    return weights, scales, router


def test_repack_preserves_all_int8_codes_scales_and_native_fp32(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    weights, scales, router = source_checkpoint(source)
    report = repack_w8a16(source, output)
    result = load_file(output / 'model-00001-of-00001.safetensors')
    assert torch.equal(unpack_uint8b128_int32(result['model.q_proj.weight_packed']), weights)
    assert torch.equal(result['model.q_proj.weight_scale'], scales)
    assert result['model.gate.weight'].dtype == torch.float32
    assert torch.equal(result['model.gate.weight'], router)
    assert report['requantized'] is False
    assert report['codes_verified'] == 1 and report['fp32_verified'] == 3
    config = json.loads((output / 'config.json').read_text())['quantization_config']
    assert config['format'] == 'pack-quantized'
    assert config['config_groups']['int8']['input_activations'] is None
    assert config['config_groups']['int8']['weights']['group_size'] == -1
    assert 'model.q_proj.weight' in load_file(source / 'model-00001-of-00001.safetensors')
    validation = validate_artifact(output)
    assert validation['valid'], validation['errors']


def test_repack_rejects_existing_or_nested_destination(tmp_path):
    source = tmp_path / 'source'
    source_checkpoint(source)
    with pytest.raises(ValueError, match='non-nested'):
        repack_w8a16(source, source / 'output')
    output = tmp_path / 'output'
    output.mkdir()
    with pytest.raises(FileExistsError):
        repack_w8a16(source, output)


def test_repack_dry_run_does_not_create_destination(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source_checkpoint(source)
    assert repack_w8a16(source, output, dry_run=True)['status'] == 'planned'
    assert not output.exists()


def test_repack_rejects_nonfinite_channel_scales(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source_checkpoint(source)
    shard = source / 'model-00001-of-00001.safetensors'
    values = load_file(shard)
    values['model.q_proj.weight_scale'][0] = float('nan')
    save_file(values, shard)
    with pytest.raises(ValueError, match='finite positive FP32'):
        repack_w8a16(source, output)
    assert json.loads((output / 'repack_report.json').read_text())['status'] == 'failed'


def test_repack_rejects_missing_unquantized_index_tensor_before_writing(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source_checkpoint(source)
    index_path = source / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    index['weight_map']['model.missing_norm.weight'] = 'model-00001-of-00001.safetensors'
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match='Shard/index tensor mismatch'):
        repack_w8a16(source, output)
    assert not output.exists()
