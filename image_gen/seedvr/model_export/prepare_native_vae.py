# SPDX-License-Identifier: Apache-2.0
"""Extract trained native VAE weights and architecture from the video ONNX pair.

No checkpoint conversion or weight folding: conv weights become KTRSC, MatMul
weights stay CK (equivalent to a channels-last 1x1 convolution).
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest={'version':1,'sources':{},'weights':{},'components':{}}
    for kind in ('encoder','decoder'):
        path=source/f'seedvr2_video_vae_{kind}.onnx'
        model=onnx.load(path)
        tensors={x.name:numpy_helper.to_array(x) for x in model.graph.initializer}
        prefix=f'vae.{kind}.'
        external=onnx.load(path,load_external_data=False)
        files={e.value for x in external.graph.initializer for e in x.external_data if e.key=='location'}
        manifest['sources'][kind]={'path':str(path.resolve()),'sha256':digest(path),'external':{f:digest(path.parent/f) for f in files}}
        def weight(name, array=None, shape=None):
            if name in manifest['weights']:return name
            a=tensors[name] if array is None else array
            if a.dtype!=np.float16:raise ValueError(f'Expected FP16: {name} {a.dtype}')
            dims=list(a.shape) if shape is None else shape
            if a.ndim==5:a=a.transpose(0,2,3,4,1)
            filename=f'weight_{len(manifest["weights"]):04d}.bin'
            a=np.ascontiguousarray(a);a.tofile(output/filename)
            manifest['weights'][name]={'file':filename,'shape':dims,'bytes':a.nbytes,'sha256':digest(output/filename)}
            return name
        convs={n.input[1]:n for n in model.graph.node if n.op_type=='Conv'}
        used_convs=set()
        assert sum(n.op_type=='Attention' for n in model.graph.node)==1, 'Expected one native Attention operator'
        def conv(name):
            key=prefix+name+'.weight';n=convs[key];attrs={a.name:onnx.helper.get_attribute_value(a) for a in n.attribute}
            used_convs.add(key)
            assert attrs.get('group',1)==1 and attrs['dilations']==[1,1,1]
            return {'op':'conv','name':name,'weight':weight(key),'bias':weight(n.input[2]),'stride':attrs['strides'],'pads':attrs['pads']}
        def norm(name,silu=True):
            return {'op':'norm','weight':weight(prefix+name+'.weight'),'bias':weight(prefix+name+'.bias'),'silu':silu}
        def res(name):
            ops=[{'op':'save'},norm(name+'.norm1'),conv(name+'.conv1'),norm(name+'.norm2'),conv(name+'.conv2')]
            shortcut=conv(name+'.conv_shortcut') if prefix+name+'.conv_shortcut.weight' in convs else None
            return ops+[{'op':'add','shortcut':shortcut}]
        def attention():
            name='mid_block.attentions.0.'
            ops=[{'op':'save'},norm(name+'group_norm',False)]
            mats=[n for n in model.graph.node if n.op_type=='MatMul']
            assert len(mats)==4
            projections=[]
            for label,node in zip(('to_q','to_k','to_v','to_out.0'),mats):
                a=tensors[node.input[1]]
                # K,C,1,1,1 -> K,1,1,1,C on disk.
                arr=a.T[:,:,None,None,None]
                projections.append({'op':'conv','name':name+label,'weight':weight(prefix+name+label+'.weight',arr),'bias':weight(prefix+name+label+'.bias'),'stride':[1,1,1],'pads':[0]*6})
            return ops+[{'op':'attention','projections':projections},{'op':'add','shortcut':None}]
        ops=[conv('conv_in')]
        if kind=='encoder':
            for i in range(4):
                for j in range(2):ops+=res(f'down_blocks.{i}.resnets.{j}')
                if i<3:ops.append(conv(f'down_blocks.{i}.downsamplers.0.conv'))
        ops+=res('mid_block.resnets.0')+attention()+res('mid_block.resnets.1')
        if kind=='decoder':
            for i in range(4):
                for j in range(3):ops+=res(f'up_blocks.{i}.resnets.{j}')
                if i<3:
                    ops += [conv(f'up_blocks.{i}.upsamplers.0.upscale_conv'),{'op':'shuffle','temporal':2 if i<2 else 1},conv(f'up_blocks.{i}.upsamplers.0.conv')]
        ops += [norm('conv_norm_out'),conv('conv_out')]
        assert used_convs==set(convs), 'Unexpected convolution architecture; rediscover the graph before extracting'
        manifest['components'][kind]=ops
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Prepared {len(manifest["weights"])} tensors in {output}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();prepare(args.source,args.output)
