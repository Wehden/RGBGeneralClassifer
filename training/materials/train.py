"""Container batch jobs: cache features, train heads/backbone, evaluate and export."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from transformers import AutoModel, AutoProcessor

LABELS=['plastic','metal','glass','wood','paper','cardboard','fabric','rubber',
        'ceramic','leather','stone','liquid','human skin','food','unknown']
REJECTIONS=['human skin','food','unknown']
MODEL='google/siglip2-base-patch16-384'
REVISION='f775b65a79762255128c981547af89addcfe0f88'
SEED=20261028


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    temporary=path.with_suffix('.temporary')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def setup():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_image(path,size,processor):
    with Image.open(path) as source:
        image=source.convert('RGB').resize((size,size),Image.Resampling(processor['resample']))
    values=np.asarray(image,dtype=np.float32)*float(processor['rescale_factor'])
    values=(values-np.asarray(processor['image_mean'],np.float32))/np.asarray(processor['image_std'],np.float32)
    return np.ascontiguousarray(values.transpose(2,0,1))


def load_rows(data,with_labels=False):
    camera=json.loads((data/'dataset.json').read_text())['entries']
    annotations={}
    if with_labels:
        label_document=json.loads((data/'teacher-labels.json').read_text())
        if label_document['dataset_sha256']!=digest(data/'dataset.json'):
            raise ValueError('Label dataset identity mismatch')
        annotations={row['id']:row['teacher_consensus'] for row in label_document['entries']}
    rows=[]
    for row in camera:
        ref=annotations.get(row['id'],{'material':None,'quality':'pending','weight':0})
        rows.append({**row,'material':ref['material'],'weight':ref['weight'],
                     'quality':ref['quality'],'source_kind':'camera'})
    rows+=json.loads((data/'controls.json').read_text())['entries']
    for row in rows:
        path=(data/row['file']).resolve()
        if not path.is_relative_to(data.resolve()) or digest(path)!=row['sha256']:
            raise ValueError('Dataset pixel identity mismatch')
        row['target']=LABELS.index(row['material']) if row['material'] is not None else -1
    if len({row['id'] for row in rows})!=len(rows):raise ValueError('Duplicate sample identity')
    return rows


def source_model(offline=False):
    source=MODEL
    if offline:
        # AutoProcessor may probe optional files on "main" even when its caller
        # passes local_files_only. An absolute pinned snapshot prevents that.
        from huggingface_hub import snapshot_download
        source=snapshot_download(MODEL,revision=REVISION,local_files_only=True)
    model=AutoModel.from_pretrained(source,revision=REVISION,attn_implementation='eager',
                                    local_files_only=offline).eval()
    processor=AutoProcessor.from_pretrained(source,revision=REVISION,use_fast=False,
                                            local_files_only=offline)
    config=processor.image_processor.to_dict()
    if config.get('do_center_crop') or config['resample'] not in [2,3]:
        raise ValueError('Unsupported source preprocessing')
    return model,processor,config


def features(vision,images,device,batch_size=24):
    arrays=[]
    vision.eval()
    with torch.inference_mode():
        for start in range(0,len(images),batch_size):
            pixels=torch.from_numpy(images[start:start+batch_size]).to(device)
            embedded=vision(pixel_values=pixels).pooler_output
            arrays.append(torch.nn.functional.normalize(embedded,dim=-1).float().cpu().numpy())
    return np.concatenate(arrays)


def cache_features(data,output,device):
    rows=load_rows(data)
    model,processor,config=source_model()
    size=config['size']['height']
    model.to(device)
    prompts=[f'This is a photo of an object made of {label}.' for label in LABELS[:12]]+[
        'This is a photo of a person or human skin.','This is a photo of food.',
        'This is a blurry photo of an unidentifiable object.']
    tokens=processor(text=prompts,padding='max_length',
                     max_length=model.config.text_config.max_position_embeddings,return_tensors='pt')
    with torch.inference_mode():
        text=model.get_text_features(**{key:value.to(device) for key,value in tokens.items()})
        text=torch.nn.functional.normalize(text,dim=-1).float().cpu().numpy()
    images=np.stack([prepare_image(data/row['file'],size,config) for row in rows])
    embedded=features(model.vision_model,images,device)
    np.savez_compressed(output/'features.npz',features=embedded,ids=np.array([row['id'] for row in rows]))
    np.save(output/'text_embeddings.npy',text,allow_pickle=False)
    write(output/'feature-manifest.json',{'model':MODEL,'revision':REVISION,'model_type':model.config.model_type,
          'processor':config,'input_size':size,'embedding_size':embedded.shape[1],
          'dataset_sha256':digest(data/'dataset.json'),'controls_sha256':digest(data/'controls.json'),
          'images':len(rows),'device':str(device),'torch':torch.__version__,'features_sha256':digest(output/'features.npz'),
          'image_sha256':[row['sha256'] for row in rows]})
    print(json.dumps({'stage':'features','images':len(rows),'shape':list(embedded.shape),'device':str(device)}),flush=True)


def softmax(logits,temperature=1):
    values=logits/temperature
    values=values-values.max(axis=1,keepdims=True)
    values=np.exp(values)
    return values/values.sum(axis=1,keepdims=True)


def metrics(logits,rows,split,temperature=1,threshold=1,margin=.05):
    values=softmax(logits,temperature)
    top=values.argmax(axis=1)
    ordered=np.sort(values,axis=1)
    accepted=(values.max(axis=1)>=threshold)&((ordered[:,-1]-ordered[:,-2])>=margin)
    accepted&=np.array([LABELS[index] not in REJECTIONS for index in top])
    support=Counter(row['material'] for row in rows if row['split']=='train' and row['target']>=0)
    accepted&=np.array([support[LABELS[index]]>=3 for index in top])
    if threshold>=1:accepted[:]=False
    result={}
    for kind in ['MINC','camera']:
        ids=[i for i,row in enumerate(rows) if row['split']==split and row['source_kind']==kind]
        labeled=[i for i in ids if rows[i]['target']>=0]
        target=[i for i in labeled if rows[i]['material'] not in REJECTIONS]
        accepted_ref=[i for i in labeled if accepted[i]]
        correct=sum(top[i]==rows[i]['target'] for i in target)
        result[kind]={'examples':len(ids),'labeled_targets':len(target),
                      'correct_top1':int(correct),'top1_agreement':correct/len(target) if target else None,
                      'accepted':sum(bool(accepted[i]) for i in ids),
                      'accepted_with_reference':len(accepted_ref),
                      'accepted_reference_correct':int(sum(top[i]==rows[i]['target'] for i in accepted_ref)),
                      'accepted_reference_agreement':sum(top[i]==rows[i]['target'] for i in accepted_ref)/len(accepted_ref)
                      if accepted_ref else None,
                      'acceptance_coverage':sum(bool(accepted[i]) for i in ids)/len(ids) if ids else None,
                      'unreferenced_accepted':sum(bool(accepted[i]) for i in ids if rows[i]['target']<0)}
    return result


def objective(logits,rows):
    result=metrics(logits,rows,'validation')
    public=result['MINC']['top1_agreement'] or 0
    camera=result['camera']
    return .7*public+.3*camera['top1_agreement'] if camera['labeled_targets']>=10 else public


def calibrate(logits,rows):
    selected=[i for i,row in enumerate(rows) if row['split']=='validation' and row['target']>=0]
    targets=np.array([rows[i]['target'] for i in selected])
    temperature=min([.5,.75,1,1.5,2],key=lambda t:
                    -np.log(np.maximum(softmax(logits[selected],t)[np.arange(len(selected)),targets],1e-9)).mean())
    threshold=1.0;coverage=-1
    for candidate in np.arange(.5,1,.025):
        result=metrics(logits,rows,'validation',temperature,float(candidate))
        cohort=[result[kind] for kind in ['MINC','camera'] if result[kind]['labeled_targets']>=10]
        if not cohort:continue
        if any(row['accepted_reference_agreement'] is None or row['accepted_reference_agreement']<.85 for row in cohort):
            continue
        if sum(row['accepted_with_reference'] for row in cohort)<10:continue
        score=sum(row['accepted_with_reference'] for row in cohort)
        if score>coverage:threshold=float(candidate);coverage=score
    return {'temperature':float(temperature),'minimum_probability':threshold,'minimum_margin':.05,
            'selection':'Maximum validation reference coverage subject to >=85% agreement per evaluable domain',
            'qualified_for_acceptance':coverage>=0}


def export_artifact(vision,head,config,text,calibration,directory,training,parity_images):
    import openvino as ov
    directory.mkdir(parents=True,exist_ok=True)
    vision=vision.cpu().float().eval()
    class Vision(nn.Module):
        def __init__(self,model):super().__init__();self.model=model
        def forward(self,pixels):
            return torch.nn.functional.normalize(self.model(pixel_values=pixels).pooler_output,dim=-1)
    wrapper=Vision(vision).eval()
    size=config['size']['height']
    converted=ov.convert_model(wrapper,example_input=torch.zeros(1,3,size,size))
    converted.reshape([1,3,size,size])
    ov.save_model(converted,directory/'vision.xml',compress_to_fp16=False)
    np.save(directory/'text_embeddings.npy',text,allow_pickle=False)
    np.savez(directory/'classifier.npz',weight=head.weight.detach().cpu().numpy(),
             bias=head.bias.detach().cpu().numpy())
    compiled=ov.Core().compile_model(str(directory/'vision.xml'),'CPU',{'INFERENCE_PRECISION_HINT':'f32','INFERENCE_NUM_THREADS':4})
    errors=[]
    with torch.inference_mode():
        for image in parity_images:
            tensor=image[None]
            reference=wrapper(torch.from_numpy(tensor)).numpy()
            actual=compiled([tensor])[compiled.output(0)]
            errors.append(float(np.max(np.abs(reference-actual))))
    if max(errors)>1e-4:raise ValueError('Trained encoder export parity failed')
    manifest={'schema_version':1,'model':MODEL,'revision':REVISION,'license':'Apache-2.0',
              'input_size':size,'embedding_size':head.in_features,'labels':LABELS,
              'rejection_labels':REJECTIONS,'prompts':training['prompts'],
              'processor':config,'minimum_similarity':.20,'minimum_margin':.02,
              'files':{name:digest(directory/name) for name in
                       ['vision.xml','vision.bin','text_embeddings.npy','classifier.npz']},
              'classifier':{'kind':'linear_softmax','file':'classifier.npz',**calibration},
              'training':{key:value for key,value in training.items() if key!='prompts'},
              'export_parity':{'images':len(errors),'maximum_embedding_absolute_error':max(errors),'gate':1e-4}}
    write(directory/'manifest.json',manifest)


def train(data,output,device,epochs=6):
    rows=load_rows(data,True)
    cached=np.load(output/'features.npz',allow_pickle=False)
    if cached['ids'].tolist()!=[row['id'] for row in rows]:raise ValueError('Feature/sample alignment mismatch')
    feature_manifest=json.loads((output/'feature-manifest.json').read_text())
    if feature_manifest['features_sha256']!=digest(output/'features.npz'):
        raise ValueError('Cached feature checksum changed')
    if feature_manifest['dataset_sha256']!=digest(data/'dataset.json') or feature_manifest['controls_sha256']!=digest(data/'controls.json'):
        raise ValueError('Feature dataset changed')
    embedded=cached['features']
    # Additional appearance filtering is independent of teacher labels and test outcomes.
    heldout=[i for i,row in enumerate(rows) if row['source_kind']=='camera' and row['split']!='train']
    purged=[]
    for i,row in enumerate(rows):
        if row['source_kind']=='camera' and row['split']=='train' and heldout:
            if float((embedded[heldout]@embedded[i]).max())>=.985:
                row['split']='excluded-near-duplicate';purged.append(row['id'])
    train_ids=[i for i,row in enumerate(rows) if row['split']=='train' and row['target']>=0]
    if not train_ids:raise ValueError('No training labels')
    targets=torch.tensor([rows[i]['target'] for i in train_ids],device=device)
    counts=Counter(targets.cpu().tolist())
    weights=torch.tensor([rows[i]['weight']/np.sqrt(counts[rows[i]['target']]) for i in train_ids],device=device,dtype=torch.float32)
    weights/=weights.mean()
    text=np.load(output/'text_embeddings.npy',allow_pickle=False)
    head=nn.Linear(embedded.shape[1],len(LABELS)).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(text*10).to(device));head.bias.zero_()
    train_features=torch.from_numpy(embedded[train_ids]).to(device)
    all_features=torch.from_numpy(embedded).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.02,weight_decay=.005)
    best=-1;best_state=None
    for step in range(401):
        optimizer.zero_grad()
        loss=(nn.functional.cross_entropy(head(train_features),targets,reduction='none')*weights).mean()
        loss.backward();optimizer.step()
        if step%10==0:
            with torch.inference_mode():logits=head(all_features).cpu().numpy()
            score=objective(logits,rows)
            if score>best:
                best=score;best_state={key:value.detach().cpu().clone() for key,value in head.state_dict().items()}
    head.load_state_dict(best_state)
    with torch.inference_mode():frozen_logits=head(all_features).cpu().numpy()
    frozen_state={key:value.detach().cpu().clone() for key,value in head.state_dict().items()}
    print(json.dumps({'stage':'frozen-head','validation_objective':best}),flush=True)
    model,processor,config=source_model(offline=True)
    vision=model.vision_model.to(device)
    del model
    for parameter in vision.parameters():parameter.requires_grad=False
    for layer in vision.encoder.layers[-2:]:
        for parameter in layer.parameters():parameter.requires_grad=True
    for component in [vision.post_layernorm,vision.head]:
        for parameter in component.parameters():parameter.requires_grad=True
    original={key:value.detach().cpu().clone() for key,value in vision.state_dict().items()
              if any(key.startswith(prefix) for prefix in ['encoder.layers.10.','encoder.layers.11.','post_layernorm.','head.'])}
    images=np.stack([prepare_image(data/row['file'],feature_manifest['input_size'],config) for row in rows])
    optimizer=torch.optim.AdamW([{'params':[p for p in vision.parameters() if p.requires_grad],'lr':5e-6},
                                 {'params':head.parameters(),'lr':3e-4}],weight_decay=.01)
    best_fine=-1;best_vision=best_head=None;history=[]
    for epoch in range(epochs):
        vision.train();head.train()
        order=torch.randperm(len(train_ids)).tolist();total=0
        for start in range(0,len(order),16):
            batch=order[start:start+16];ids=[train_ids[k] for k in batch]
            pixels=torch.from_numpy(images[ids]).to(device)
            if random.random()<.5:pixels=pixels.flip(-1)
            pixels=(pixels+random.uniform(-.03,.03)).clamp(-1,1)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                outputs=torch.nn.functional.normalize(vision(pixel_values=pixels).pooler_output,dim=-1)
                loss=(nn.functional.cross_entropy(head(outputs),targets[batch],reduction='none')*weights[batch]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in vision.parameters() if p.requires_grad]+list(head.parameters()),1)
            optimizer.step();total+=float(loss.detach())
        current_features=features(vision,images,device)
        with torch.inference_mode():logits=head(torch.from_numpy(current_features).to(device)).cpu().numpy()
        score=objective(logits,rows)
        history.append({'epoch':epoch+1,'loss_sum':total,'validation_objective':score})
        print(json.dumps({'stage':'fine-tune',**history[-1]}),flush=True)
        if score>best_fine:
            best_fine=score
            best_vision={key:value.detach().cpu().clone() for key,value in vision.state_dict().items()}
            best_head={key:value.detach().cpu().clone() for key,value in head.state_dict().items()}
    vision.load_state_dict(best_vision);head.load_state_dict(best_head)
    changed=sum(not torch.equal(original[key],best_vision[key]) for key in original)
    if changed==0:raise ValueError('Fine-tuning did not update encoder parameters')
    tuned_features=features(vision,images,device)
    with torch.inference_mode():tuned_logits=head(torch.from_numpy(tuned_features).to(device)).cpu().numpy()
    calibrations={'frozen-head':calibrate(frozen_logits,rows),'fine-tuned':calibrate(tuned_logits,rows)}
    evaluations={}
    for name,logits in [('frozen-head',frozen_logits),('fine-tuned',tuned_logits)]:
        calibration=calibrations[name]
        evaluations[name]={'validation':metrics(logits,rows,'validation',calibration['temperature'],calibration['minimum_probability']),
                           'test':metrics(logits,rows,'test',calibration['temperature'],calibration['minimum_probability']),
                           'calibration':calibration,'validation_objective':best if name=='frozen-head' else best_fine}
    base_similarities=embedded@text.T
    baseline_top=base_similarities.argmax(axis=1)
    # Preserve the original zero-shot acceptance operating point.
    baseline={}
    for kind in ['MINC','camera']:
        ids=[i for i,row in enumerate(rows) if row['split']=='test' and row['source_kind']==kind and row['target']>=0
             and row['material'] not in REJECTIONS]
        baseline[kind]={'references':len(ids),'top1_correct':int(sum(baseline_top[i]==rows[i]['target'] for i in ids)),
                        'top1_agreement':sum(baseline_top[i]==rows[i]['target'] for i in ids)/len(ids) if ids else None}
    preferred=max(evaluations,key=lambda key:evaluations[key]['validation_objective'])
    training={'labels_sha256':digest(data/'teacher-labels.json'),'dataset_sha256':digest(data/'dataset.json'),
              'controls_sha256':digest(data/'controls.json'),'seed':SEED,'epochs':epochs,
              'train_examples':len(train_ids),'camera_train_examples':sum(rows[i]['source_kind']=='camera' for i in train_ids),
              'train_material_counts':dict(Counter(rows[i]['material'] for i in train_ids)),
              'purged_camera_near_duplicates':purged,'encoder_tensors_changed':changed,
              'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
              'torch':torch.__version__,'source_model_type':feature_manifest['model_type'],
              'source_revision':REVISION,'prompts':[f'This is a photo of an object made of {label}.' for label in LABELS[:12]]+
              ['This is a photo of a person or human skin.','This is a photo of food.',
               'This is a blurry photo of an unidentifiable object.']}
    summary={'training':{key:value for key,value in training.items() if key!='prompts'},'evaluations':evaluations,
             'zero_shot_test':baseline,'preferred_by_validation':preferred,'fine_tune_history':history,
             'scope':'Camera labels are weak VLM supervision; MINC labels are dataset annotations'}
    write(output/'results.json',summary)
    np.savez_compressed(output/'evaluation-logits.npz',frozen=frozen_logits,tuned=tuned_logits,
                        zero_shot_similarities=base_similarities,ids=cached['ids'])
    write(output/'evaluation-rows.json',rows)
    frozen_head=nn.Linear(embedded.shape[1],len(LABELS));frozen_head.load_state_dict(frozen_state)
    source,_,_=source_model(offline=True)
    size=feature_manifest['input_size']
    parity=list(images[:6])+[np.zeros((3,size,size),np.float32),
                             np.stack([np.ones((size,size),np.float32),
                                       -np.ones((size,size),np.float32),-np.ones((size,size),np.float32)]),
                             np.random.default_rng(SEED).uniform(-1,1,(3,size,size)).astype(np.float32)]
    export_artifact(source.vision_model,frozen_head,config,text,calibrations['frozen-head'],
                    output/'artifacts'/'frozen-head',{**training,'variant':'frozen-head','encoder_tensors_changed':0},parity)
    export_artifact(vision,head,config,text,calibrations['fine-tuned'],
                    output/'artifacts'/'fine-tuned',{**training,'variant':'fine-tuned'},parity)
    print(json.dumps({'stage':'complete','preferred_by_validation':preferred,'evaluations':evaluations}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['features','train'],required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=6)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    device=setup()
    started=time.monotonic()
    if args.stage=='features':cache_features(args.data,args.output,device)
    else:train(args.data,args.output,device,args.epochs)
    print(json.dumps({'elapsed_s':time.monotonic()-started,'device':str(device)}),flush=True)
