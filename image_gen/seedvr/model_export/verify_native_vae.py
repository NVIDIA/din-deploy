# SPDX-License-Identifier: Apache-2.0
"""Compare native encoder and decoder against unmodified source ONNX, FP16 I/O."""
import argparse,json,subprocess,hashlib,shutil
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

def compare(expected,actual):
    a=expected.astype(np.float32);b=actual.astype(np.float32)
    diff=np.abs(a-b);finite=bool(np.isfinite(a).all() and np.isfinite(b).all());close=np.isclose(a,b,atol=.01,rtol=.01)
    return dict(finite=finite,max_abs=float(diff.max()),mean_abs=float(diff.mean()),failed_elements=int((~close).sum()),elements=a.size,passed=bool(finite and close.all()),atol=.01,rtol=.01)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','weights','exe','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--image',action='store_true',help='Check single-frame execution against the image ONNX exports.')
    p.add_argument('--reference-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    opts=ort.SessionOptions();opts.intra_op_num_threads=8
    summary={}
    for kind in ('encoder','decoder'):
        model_path=a.source/f'seedvr2_{"ema" if a.image else "video"}_vae_{kind}.onnx'
        with model_path.open('rb') as f:model_hash=hashlib.file_digest(f,'sha256').hexdigest()
        graph=onnx.load(model_path,load_external_data=False)
        external_hashes={}
        for filename in {e.value for v in graph.graph.initializer for e in v.external_data if e.key=='location'}:
            relative=Path(filename)
            if relative.is_absolute() or relative.name!=filename:raise ValueError('Expected adjacent external data')
            with (model_path.parent/relative).open('rb') as f:external_hashes[filename]=hashlib.file_digest(f,'sha256').hexdigest()
        terminal=next(n for n in graph.graph.node if graph.graph.output[0].name in n.output)
        clipped=terminal.op_type=='Clip'
        if clipped:
            values={v.name:numpy_helper.to_array(v) for v in graph.graph.initializer if v.name in terminal.input[1:]}
            assert float(values[terminal.input[1]])==0 and float(values[terminal.input[2]])==1
            preclip=terminal.input[0]
        session=None
        for h,w in ((64,64),(72,128)):
            case=a.output/f'{kind}_{h}x{w}';case.mkdir(exist_ok=True)
            rng=np.random.default_rng(42)
            x=rng.random((1,3,1 if a.image else 5,h,w)).astype(np.float16) if kind=='encoder' else rng.standard_normal((1,16,1 if a.image else 2,h//8,w//8)).astype(np.float16)
            x.tofile(case/'input.bin')
            provenance={'model':str(model_path.resolve()),'sha256':model_hash,'external_sha256':external_hashes,'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'onnxruntime':ort.__version__,'provider':'CPUExecutionProvider','terminal_clip_0_1':clipped}
            metadata=case/'reference_provenance.json'
            if not (case/'reference.npy').exists() or not metadata.exists() or json.loads(metadata.read_text())!=provenance:
                if session is None:
                    if clipped:
                        # ONNX rejects multiply hard-linked external data. Use
                        # ordinary private copies; retain the source untouched.
                        copied=a.output/'reference_source';copied.mkdir(exist_ok=True)
                        shutil.copyfile(model_path,copied/model_path.name)
                        for filename in {e.value for v in graph.graph.initializer for e in v.external_data if e.key=='location'}:
                            relative=Path(filename)
                            if relative.is_absolute() or relative.name!=filename:raise ValueError('Expected adjacent external data')
                            shutil.copyfile(model_path.parent/relative,copied/relative)
                        full=onnx.load(copied/model_path.name)
                        extra=onnx.ValueInfoProto();extra.CopyFrom(full.graph.output[0]);extra.name=preclip;full.graph.output.append(extra)
                        session=ort.InferenceSession(full.SerializeToString(),opts,providers=['CPUExecutionProvider']);del full
                    else:session=ort.InferenceSession(str(model_path),opts,providers=['CPUExecutionProvider'])
                ys=session.run(None,{('image' if kind=='encoder' else 'latent'):x.squeeze(2) if a.image else x});np.save(case/'reference.npy',ys[0])
                if clipped:np.save(case/'reference_preclip.npy',ys[1])
                metadata.write_text(json.dumps(provenance,indent=2))
                print('Reference complete',kind,h,w,flush=True)
            if a.reference_only:continue
            cmd=[str(a.exe),str(a.weights),'encode' if kind=='encoder' else 'decode',*map(str,x.shape[2:]),str(case/'input.bin'),str(case/'actual.bin'),str(case/'runtime.json')]
            (case/'command.json').write_text(json.dumps(cmd,indent=2))
            with open(case/'run.log','w') as log:run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if run.returncode:raise RuntimeError(f'{kind} native execution failed: {case / "run.log"}')
            expected=np.load(case/'reference.npy');actual=np.fromfile(case/'actual.bin',np.float16).reshape(expected.shape)
            result=compare(expected,np.clip(actual,0,1) if clipped else actual)
            if clipped:
                result['comparison']='Native output after pipeline [0,1] clamp vs original ONNX output'
                result['native_raw_vs_onnx_preclip']=compare(np.load(case/'reference_preclip.npy'),actual)
                result['passed']=result['passed'] and result['native_raw_vs_onnx_preclip']['passed']
            summary[case.name]=result
            (case/'accuracy.json').write_text(json.dumps(result,indent=2));(a.output/'accuracy.json').write_text(json.dumps(summary,indent=2));print(case.name,result,flush=True)
        del session
    if summary and not all(r['passed'] for r in summary.values()):raise SystemExit('Numerical parity failed; tolerances unchanged')

if __name__=='__main__':main()
