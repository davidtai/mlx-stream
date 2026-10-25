"""CPU-only OQ3 conversion: python3 -m unittest discover -s scripts -p test_prepare_minimax_oq3_pack.py -v"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("prepare_minimax_oq3_pack", Path(__file__).with_name("prepare_minimax_oq3_pack.py"))
pack = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pack
SPEC.loader.exec_module(pack)
HIDDEN, INTER, EXPERTS, FIRST, LAYERS = 128, 64, 3, 1, 3
PROJS = ("gate_proj", "up_proj", "down_proj")
WIDTHS = {"U32": 4, "BF16": 2, "F32": 4, "U8": 1}


def prod(shape):
    n = 1
    for value in shape:
        n *= value
    return n


def safetensors(path, tensors):
    header, data = {}, bytearray()
    for name, (dtype, shape, value) in tensors.items():
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[len(data), len(data) + len(value)])
        data.extend(value)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    return header


def fixture(path):
    path.mkdir()
    quant = dict(bits=3, group_size=64, mode="affine")
    cfg = dict(model_type="minimax_m3_vl", text_config=dict(hidden_size=HIDDEN, intermediate_size=INTER,
        dense_intermediate_size=256, shared_intermediate_size=INTER, num_local_experts=EXPERTS,
        num_hidden_layers=LAYERS, num_experts_per_tok=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=64, vocab_size=16, pack_shared_expert=False, routed_expert_scales=True,
        moe_layer_freq=[0, 1, 1], sparse_attention_config=dict(sparse_index_dim=64,sparse_num_index_heads=1)))
    tensors = {}
    for layer in range(FIRST, LAYERS):
        for p, projection in enumerate(PROJS):
            out, inn = (HIDDEN, INTER) if p == 2 else (INTER, HIDDEN)
            base = f"language_model.model.layers.{layer}.block_sparse_moe.switch_mlp.{projection}"
            for suffix, dtype, shape in (("weight", "U32", [EXPERTS, out, inn * 3 // 32]),
                                         ("scales", "BF16", [EXPERTS, out, inn // 64]),
                                         ("biases", "BF16", [EXPERTS, out, inn // 64])):
                n = prod(shape) * WIDTHS[dtype]
                data = bytes((j * 19 + layer * 11 + p * 7) % 256 for j in range(n))
                tensors[base + "." + suffix] = dtype, shape, data
    def linear(base, out, inn, bits, mode="affine"):
        group = 32 if mode == "mxfp8" else 64
        quant[base] = dict(bits=bits, group_size=group, mode=mode)
        tensors[base + ".weight"] = "U32", [out, inn * bits // 32], bytes((j * 13 + bits) % 256 for j in range(out * inn * bits // 8))
        dtype = "U8" if mode == "mxfp8" else "BF16"
        tensors[base + ".scales"] = dtype, [out, inn // group], bytes([127]) * (out * inn // group * WIDTHS[dtype])
        if mode == "affine":
            tensors[base + ".biases"] = "BF16", [out, inn // group], b"\x80\xbf" * (out * inn // group)
    linear("language_model.model.embed_tokens", 16, HIDDEN, 8)
    linear("language_model.lm_head", 16, HIDDEN, 8)
    linear("language_model.model.layers.0.self_attn.q_proj", 128, HIDDEN, 4)
    linear("language_model.model.layers.0.self_attn.k_proj", 64, HIDDEN, 5)
    linear("language_model.model.layers.0.self_attn.o_proj", HIDDEN, HIDDEN, 3)
    linear("language_model.model.layers.0.mlp.up_proj", 256, HIDDEN, 6)
    linear("language_model.model.layers.1.block_sparse_moe.shared_experts.gate_proj", INTER, HIDDEN, 8, "mxfp8")
    tensors["language_model.model.norm.weight"] = "BF16", [HIDDEN], b"\0\0" * HIDDEN
    tensors["language_model.model.layers.1.block_sparse_moe.gate.weight"] = "F32", [EXPERTS,HIDDEN], b"\0\0\x80\x3f" * (EXPERTS * HIDDEN)
    tensors["language_model.model.layers.1.block_sparse_moe.e_score_correction_bias"] = "F32", [EXPERTS], b"\0" * (EXPERTS*4)
    tensors["vision_tower.weight"] = "F32", [1], b"\0\0\0\0"
    cfg["quantization"] = quant
    cfg["quantization_config"] = copy.deepcopy(quant)
    (path / "config.json").write_text(json.dumps(cfg))
    headers, wm = {}, {}
    for shard in range(2):
        filename = f"model-{shard}.safetensors"
        subset = {n:t for number,(n,t) in enumerate(tensors.items()) if number%2==shard}
        headers[filename] = safetensors(path / filename, subset)
        wm.update({n:filename for n in subset})
    index = dict(metadata=dict(total_size=sum(len(t[2]) for t in tensors.values())), weight_map=wm)
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    (path / "tokenizer.json").write_text('{}')
    return cfg, tensors, index, headers


class Oq3Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.src,self.out = self.root / "source",self.root / "pack"
        self.cfg,self.tensors,self.index,self.headers = fixture(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        return pack.prepare(self.src, self.out, source_repo="test/oq3", source_revision="a"*40)

    def test_bank_preserves_every_expert_weight_scale_bias_and_padding(self):
        m = self.build()
        bank = (self.out / "experts.bin").read_bytes()
        self.assertEqual((m["format"],m["layout"]), ("mlx-stream-affine-expert-bank-v1","affine-gud-9part"))
        self.assertEqual(m["quantization"], dict(mode="affine",bits=3,group_size=64))
        self.assertEqual(len(m["records"]),6)
        for number, record in enumerate(m["records"]):
            self.assertEqual(record["offset"],number*m["record_stride"])
            logical = bank[record["offset"]:record["offset"]+record["length"]]
            expected = []
            for part in m["parts"]:
                name = f"language_model.model.layers.{record['layer']}.block_sparse_moe.switch_mlp.{part['name']}"
                data = self.tensors[name][2]
                begin = record['expert'] * part['length']
                piece = data[begin:begin+part['length']]
                expected.append(piece)
                self.assertEqual(logical[part['offset']:part['offset']+part['length']],piece)
            self.assertEqual(logical,b''.join(expected))
            self.assertEqual(record['sha256'],hashlib.sha256(logical).hexdigest())
            pad = bank[record['offset']+record['length']:(number+1)*m['record_stride']]
            self.assertEqual(pad,bytes(len(pad)))
        self.assertEqual(m['output_files']['experts.bin']['sha256'],hashlib.sha256(bank).hexdigest())
        receipt = pack.common.json_read(self.out / 'pack-manifest.json')
        for name,(_,_,data) in self.tensors.items():
            if '.switch_mlp.' in name:
                self.assertEqual(receipt['source_expert_tensor_sha256'][name],hashlib.sha256(data).hexdigest())

    def test_mixed_resident_tensors_and_quantization_are_preserved_exactly(self):
        self.build()
        result = pack.common.read_header(self.out / 'model.safetensors')
        for name,(dtype,shape,data) in self.tensors.items():
            if not name.startswith('language_model.') or '.switch_mlp.' in name:continue
            target = name.removeprefix('language_model.')
            self.assertEqual((result[target].dtype,result[target].shape),(dtype,tuple(shape)))
            self.assertEqual(b''.join(result[target].chunks()),data)
        self.assertFalse(any('.switch_mlp.' in n or 'vision_tower' in n for n in result))
        cfg = pack.common.json_read(self.out / 'config.json')
        self.assertEqual(cfg['residents_format'],'mixed')
        self.assertEqual(cfg['expert_streaming']['format'],'mlx-stream-affine-expert-bank-v1')
        for name,spec in self.cfg['quantization'].items():
            self.assertEqual(cfg['quantization'][name.removeprefix('language_model.')],spec)
        self.assertEqual(cfg['quantization_config'],cfg['quantization'])
        self.assertEqual((self.out/'source-config.json').read_bytes(),(self.src/'config.json').read_bytes())
        self.assertEqual((self.out/'source-index.json').read_bytes(),(self.src/'model.safetensors.index.json').read_bytes())
        self.assertFalse((self.out/'source-residents.safetensors').exists())

    def test_header_only_estimate_does_not_read_tensor_payloads(self):
        with mock.patch.object(pack.common.Tensor,'chunks',side_effect=AssertionError('no payload reads')):
            estimate = pack.inspect_headers(self.cfg,self.index,self.headers)
        g = pack.geometry(self.cfg)
        self.assertEqual(estimate['bank_bytes'],g['record_stride']*6)
        resident = sum(len(t[2]) for n,t in self.tensors.items() if n.startswith('language_model.') and '.switch_mlp.' not in n)
        self.assertEqual(estimate['resident_bytes'],resident)
        self.assertEqual(estimate['minimum_output_bytes'],estimate['bank_bytes']+resident)

    def test_released_bank_geometry_and_disk_estimate(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['text_config'].update(hidden_size=6144,intermediate_size=3072,num_local_experts=128,
            num_hidden_layers=60,moe_layer_freq=[0]*3+[1]*57)
        g=pack.geometry(cfg)
        self.assertEqual(g['record_bytes'],24772608)
        self.assertEqual(g['record_stride'],24772608)
        self.assertEqual(g['gate_up_bytes'],16515072)
        self.assertEqual(g['down_bytes'],8257536)
        self.assertEqual(g['record_stride']*128*57,180740947968)

    def test_incomplete_or_mistyped_expert_components_refuse_before_output(self):
        bad=copy.deepcopy(self.headers)
        name=next(n for header in bad.values() for n in header if '.switch_mlp.' in n and n.endswith('.scales'))
        shard=self.index['weight_map'][name]
        bad[shard][name]['dtype']='F32'
        with self.assertRaisesRegex(pack.common.PrepError,'expert.*layout|expert.*dtype'):
            pack.inspect_headers(self.cfg,self.index,bad)
        bad=copy.deepcopy(self.headers)
        del bad[shard][name]
        with self.assertRaisesRegex(pack.common.PrepError,'index|missing'):
            pack.inspect_headers(self.cfg,self.index,bad)
        self.assertFalse(self.out.exists())

    def test_changed_expert_quantization_and_unknown_side_tensors_refuse(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['quantization']['bits']=4
        cfg['quantization_config']=copy.deepcopy(cfg['quantization'])
        with self.assertRaisesRegex(pack.common.PrepError,'affine.*3|3.*affine'):
            pack.geometry(cfg)
        bad=copy.deepcopy(self.headers)
        name=next(n for header in bad.values() for n in header if '.switch_mlp.' in n)
        shard=self.index['weight_map'][name]
        other=name.rsplit('.',1)[0]+'.global_scale'
        bad[shard][other]=dict(dtype='F32',shape=[],data_offsets=[0,4])
        idx=copy.deepcopy(self.index);idx['weight_map'][other]=shard
        with self.assertRaisesRegex(pack.common.PrepError,'unsupported.*expert|expert.*tensor'):
            pack.inspect_headers(self.cfg,idx,bad)

    def test_atomic_failure_and_existing_output_protection(self):
        with mock.patch.object(pack,'write_residents',side_effect=pack.common.PrepError('fixture failure')):
            with self.assertRaisesRegex(pack.common.PrepError,'fixture failure'):self.build()
        self.assertFalse(self.out.exists())
        self.assertFalse(self.out.with_name('.pack.prepare').exists())
        self.out.mkdir();(self.out/'keep').write_text('existing')
        with self.assertRaisesRegex(pack.common.PrepError,'empty'):self.build()
        self.assertEqual((self.out/'keep').read_text(),'existing')

    def test_source_payload_pins_are_checked_before_any_output(self):
        pins={file.name:(file.stat().st_size,hashlib.sha256(file.read_bytes()).hexdigest()) for file in self.src.iterdir()}
        with mock.patch.object(pack,'SOURCE_PINS',pins):
            pack.prepare(self.src,self.out)
            receipt=pack.common.json_read(self.out/'pack-manifest.json')
            self.assertEqual(receipt['source']['verification'],'official-oq3-release-sha256')
            file=self.src/'model-0.safetensors';data=file.read_bytes()
            file.write_bytes(data[:-1]+bytes([data[-1]^1]))
            other=self.root/'corrupt'
            with self.assertRaisesRegex(pack.common.PrepError,'sha256 mismatch'):
                pack.prepare(self.src,other)
            self.assertFalse(other.exists())

    def test_bounded_chunks_preserve_the_same_bank_and_residents(self):
        self.build()
        other=self.root/'small-chunks'
        with mock.patch.object(pack.common,'CHUNK',64):
            pack.prepare(self.src,other,source_repo='test/oq3',source_revision='a'*40)
        for name in ('experts.bin','model.safetensors'):
            self.assertEqual((self.out/name).read_bytes(),(other/name).read_bytes())

    def test_unknown_resident_and_wrong_projection_shape_refuse(self):
        idx=copy.deepcopy(self.index)
        name='language_model.model.layers.0.self_attn.q_proj.weight'
        shard=idx['weight_map'][name]
        changed=copy.deepcopy(self.tensors)
        removed=0
        for suffix in ('weight','scales','biases'):
            key=name.removesuffix('.weight')+'.'+suffix
            dtype,shape,data=changed[key]
            row_bytes=len(data)//shape[0]
            changed[key]=(dtype,[shape[0]-1,*shape[1:]],data[:-row_bytes])
            removed+=row_bytes
        bad={}
        for number,filename in enumerate(self.headers):
            subset={n:t for n,t in changed.items() if idx['weight_map'][n]==filename}
            bad[filename]=safetensors(self.root/f'wrong-shape-{number}.safetensors',subset)
        idx['metadata']['total_size']-=removed
        with self.assertRaisesRegex(pack.common.PrepError,'resident.*shape'):
            pack.inspect_headers(self.cfg,idx,bad)
        unknown='language_model.model.layers.0.mystery.weight'
        bad=copy.deepcopy(self.headers);idx=copy.deepcopy(self.index)
        bad[shard][unknown]=dict(dtype='F32',shape=[1],data_offsets=[0,4]);idx['weight_map'][unknown]=shard
        with self.assertRaisesRegex(pack.common.PrepError,'unsupported resident'):
            pack.inspect_headers(self.cfg,idx,bad)

    def test_isolated_cli_needs_no_heavy_packages(self):
        cmd=[sys.executable,'-I','-S',str(Path(pack.__file__).resolve()),'--src',str(self.src),'--out',str(self.out),
             '--source-repo','test/oq3','--source-revision','a'*40]
        result=subprocess.run(cmd,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('prepared 6 expert records',result.stdout)


if __name__=='__main__':unittest.main()
