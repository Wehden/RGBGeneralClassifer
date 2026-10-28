#!/usr/bin/env python3
"""Two blinded Nova passes, per-row validation and resumable private training labels."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw
from rgb.material_training import SYSTEM, label_prompt, parse_labels, consensus, protocol_hash
from rgb.vlm_providers import BedrockProvider, ReviewRequest, PRIMARY_MODEL


def write(path, value):
    temporary = path.with_suffix('.temporary')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def jpeg(image):
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=92)
    return stream.getvalue()


def prepare(directory):
    dataset = json.loads((directory/'dataset.json').read_text())
    batches = []
    evidence = directory/'teacher-evidence'
    evidence.mkdir(exist_ok=True)
    for start in range(0,len(dataset['entries']),4):
        rows = dataset['entries'][start:start+4]
        images = []
        batch_id = f'batch-{len(batches):03}'
        for index,row in enumerate(rows):
            if digest(directory/row['file']) != row['sha256'] or digest(directory/row['scene_file']) != row['scene_sha256']:
                raise ValueError('Training evidence changed')
            with Image.open(directory/row['scene_file']) as source:
                scene = source.convert('RGB')
            draw = ImageDraw.Draw(scene)
            x1,y1,x2,y2=row['bounds_xyxy']
            draw.rectangle((x1,y1,x2-1,y2-1),outline='#ff2d55',width=3)
            draw.text((x1,max(0,y1-22)),str(index),fill='#ff2d55',font_size=18)
            scene.thumbnail((768,768))
            with Image.open(directory/row['file']) as source:
                crop=source.convert('RGB');crop.thumbnail((512,512))
            for kind,image in [('scene',scene),('crop',crop)]:
                path=evidence/f'{batch_id}-{index}-{kind}.jpg'
                content=jpeg(image)
                if path.exists() and path.read_bytes()!=content:
                    raise ValueError('Teacher image encoding changed')
                path.write_bytes(content)
                images.append({'label':f'Sample {index}: '+('scene context' if kind=='scene' else 'unannotated target crop'),
                               'file':str(path.relative_to(directory)), 'sha256':digest(path)})
        batches.append({'id':batch_id,'sample_ids':[row['id'] for row in rows],
                        'prompt':label_prompt(len(rows)),'images':images})
    manifest={'dataset_sha256':digest(directory/'dataset.json'),'protocol_sha256':protocol_hash(),
              'system':SYSTEM,'model':PRIMARY_MODEL,'profile':'default','region':'us-west-2','batches':batches}
    path=directory/'teacher-manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise ValueError('Teacher protocol/dataset changed; use a new run directory')
    write(path,manifest)
    return dataset,manifest


def collect(directory,dataset,manifest):
    rows={row['id']:{**row,'teacher_reviews':[None,None]} for row in dataset['entries']}
    states=Counter();input_tokens=output_tokens=0;recovered=0
    for pass_index in range(2):
        for batch in manifest['batches']:
            path=directory/'teacher-requests'/f"{batch['id']}-pass-{pass_index+1}.json"
            if not path.exists():continue
            receipt=json.loads(path.read_text());states[receipt['status']]+=1
            input_tokens+=receipt.get('audit',{}).get('prompt_tokens') or 0
            output_tokens+=receipt.get('audit',{}).get('output_tokens') or 0
            if 'response_text' not in receipt:continue
            try:
                parsed,_=parse_labels(receipt['response_text'],len(batch['sample_ids']))
            except (ValueError,TypeError):
                continue
            recovered+=int(receipt['status']=='error' and bool(parsed))
            for index,sample_id in enumerate(batch['sample_ids']):
                rows[sample_id]['teacher_reviews'][pass_index]=parsed.get(index)
    entries=[]
    for row in rows.values():
        row.update(teacher_consensus=consensus(row['teacher_reviews']))
        entries.append(row)
    summary={'samples':len(entries),'request_states':dict(states),
             'planned_calls':2*len(manifest['batches']),
             'consensus_counts':dict(Counter(row['teacher_consensus']['quality'] for row in entries)),
             'material_counts':dict(Counter(row['teacher_consensus']['material'] for row in entries
                                            if row['teacher_consensus']['material'] is not None)),
             'input_tokens':input_tokens,'output_tokens':output_tokens,
             'formatting_recovered_requests':recovered,
             'parser_source_sha256':digest(Path(__file__).resolve().parents[1]/'rgb/material_training.py')}
    write(directory/'teacher-labels.json',{'schema':1,'label_source':'two-pass Nova weak supervision',
          'dataset_sha256':manifest['dataset_sha256'],'protocol_sha256':manifest['protocol_sha256'],
          'entries':entries,'summary':summary})
    return summary


def run(directory,invoke,maximum):
    dataset,manifest=prepare(directory)
    requests=directory/'teacher-requests';requests.mkdir(exist_ok=True)
    provider=BedrockProvider(profile='default',region='us-west-2',read_timeout=90)
    calls=0
    if invoke:
        for pass_index in range(2):
            for batch in manifest['batches']:
                path=requests/f"{batch['id']}-pass-{pass_index+1}.json"
                if path.exists():continue
                if calls>=maximum:return collect(directory,dataset,manifest)
                images=tuple((row['label'],(directory/row['file']).read_bytes()) for row in batch['images'])
                receipt={'status':'started','batch_id':batch['id'],'pass':pass_index+1,
                         'started_at':datetime.now(timezone.utc).isoformat(),
                         'model':PRIMARY_MODEL,'protocol_sha256':manifest['protocol_sha256'],
                         'prompt_sha256':hashlib.sha256(batch['prompt'].encode()).hexdigest(),
                         'image_sha256':[row['sha256'] for row in batch['images']]}
                write(path,receipt);started=time.monotonic()
                try:
                    result=provider.review(ReviewRequest(PRIMARY_MODEL,SYSTEM,batch['prompt'],images,2400))
                    receipt.update(response_text=result.text,audit=result.audit)
                    parsed,errors=parse_labels(result.text,len(batch['sample_ids']))
                    receipt.update(parsed_rows=parsed,row_errors=errors,status='partial' if errors else 'ok')
                except Exception as exc:
                    receipt.update(status='error',error_type=type(exc).__name__)
                    if hasattr(exc,'audit'):receipt['audit']=exc.audit
                    if isinstance(exc,ValueError):receipt['validation_error']=str(exc)
                receipt['elapsed_s']=time.monotonic()-started
                write(path,receipt);calls+=1
                print(f"{batch['id']} pass {pass_index+1}: {receipt['status']}",flush=True)
    return collect(directory,dataset,manifest)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--invoke-bedrock',action='store_true')
    parser.add_argument('--max-calls',type=int,default=128)
    args=parser.parse_args()
    print(json.dumps(run(args.dataset_dir,args.invoke_bedrock,args.max_calls),indent=2))
