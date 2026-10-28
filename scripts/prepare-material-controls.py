#!/usr/bin/env python3
"""Bounded MINC train/validation/test controls, excluding every earlier pilot image."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from PIL import Image


def canonical(path):
    return '/'.join(Path(path).parts[-3:])


def prepare(archive, splits_file, prior_dataset, output, train_count=100, heldout_count=20):
    output.mkdir(parents=True,exist_ok=True)
    if (output/'controls.json').exists():
        raise ValueError('Controls already exist; preserve this snapshot and use a new run directory')
    mapping={name:name for name in ['ceramic','fabric','glass','leather','metal','paper','plastic','stone','wood']}
    mapping['water']='liquid'
    previous=json.loads(prior_dataset.read_text())
    excluded={canonical(row['source_image']) for row in previous['images']}
    splits=json.loads(splits_file.read_text())
    wanted={}
    for split,filename,count in [('train','train1.txt',train_count),
                                 ('validation','validate1.txt',heldout_count),
                                 ('test','test1.txt',heldout_count)]:
        text=next(value for key,value in splits.items() if key.endswith('/'+filename))
        for category,label in mapping.items():
            names=[line for line in text.splitlines() if line.startswith('images/'+category+'/') and line not in excluded]
            names.sort(key=lambda name:hashlib.sha256(name.encode()).digest())
            if len(names)<count:raise ValueError('Insufficient MINC examples')
            for name in names[:count]:
                if name in wanted:raise ValueError('Official split overlap')
                wanted[name]={'split':split,'label':label}
    entries=[]
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            name=canonical(member.name)
            if not member.isfile() or name not in wanted:continue
            if member.size>8*1024*1024:raise ValueError('Unexpected MINC image size')
            data=tar.extractfile(member).read()
            image=Image.open(io.BytesIO(data)).convert('RGB')
            metadata=wanted[name]
            sample_id='minc-'+hashlib.sha256(name.encode()).hexdigest()[:20]
            directory=output/'minc'/metadata['split'];directory.mkdir(parents=True,exist_ok=True)
            path=directory/(sample_id+'.png');image.save(path)
            entries.append({'id':sample_id,'file':str(path.relative_to(output)),
                            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'split':metadata['split'],'material':metadata['label'],'weight':1.0,
                            'source_kind':'MINC','source_image':name,'group_id':sample_id})
    if len(entries)!=len(wanted):raise ValueError('Missing selected MINC images')
    manifest={'schema':1,'selection':'Lowest SHA256 image paths within official split 1, excluding prior pilot',
              'excluded_prior_images':len(excluded),'train_per_category':train_count,
              'validation_per_category':heldout_count,'test_per_category':heldout_count,
              'split_metadata_sha256':hashlib.sha256(splits_file.read_bytes()).hexdigest(),
              'entries':sorted(entries,key=lambda row:row['id'])}
    (output/'controls.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return {'examples':len(entries),'excluded_prior_images':len(excluded)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--splits',type=Path,required=True)
    parser.add_argument('--prior-dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.archive,args.splits,args.prior_dataset,args.output)))
