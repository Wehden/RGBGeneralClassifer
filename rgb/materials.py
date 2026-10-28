"""Offline SigLIP2 crop classification; scores are not material probabilities."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(rgb, manifest):
    processor=manifest['processor']
    if (processor.get('do_center_crop') or processor.get('resample') not in {2,3}
            or not processor.get('do_resize') or not processor.get('do_rescale')
            or not processor.get('do_normalize')):
        raise ValueError('Unsupported material image processor')
    size=manifest['input_size']
    resized=Image.fromarray(rgb).resize((size,size),Image.Resampling(processor['resample']))
    image=np.asarray(resized,dtype=np.float32)*float(processor['rescale_factor'])
    image=(image-np.asarray(processor['image_mean'],np.float32))/np.asarray(processor['image_std'],np.float32)
    return np.ascontiguousarray(image.transpose(2,0,1)[None],dtype=np.float32)


def rank(embedding, text, manifest):
    embedding=np.asarray(embedding,dtype=np.float32).reshape(-1)
    if embedding.shape!=(manifest['embedding_size'],) or not np.isfinite(embedding).all():
        raise ValueError('Invalid material embedding')
    norm=np.linalg.norm(embedding)
    if norm<1e-8:raise ValueError('Empty material embedding')
    similarities=np.clip(text @ (embedding/norm),-1,1)
    if not np.isfinite(similarities).all():raise ValueError('Invalid material scores')
    order=np.argsort(-similarities,kind='stable')
    best,second=map(int,order[:2])
    margin=float(similarities[best]-similarities[second])
    accepted=(float(similarities[best])>=manifest['minimum_similarity']
              and margin>=manifest['minimum_margin']
              and manifest['labels'][best] not in manifest.get('rejection_labels',[]))
    return {'status':'identified' if accepted else 'uncertain',
            'label':manifest['labels'][best] if accepted else None,
            'best_candidate':manifest['labels'][best],
            'similarity':round(float(similarities[best]),6),'margin':round(margin,6),
            'score_type':'uncalibrated_image_text_similarity',
            'candidates':[{'label':manifest['labels'][int(index)],
                           'similarity':round(float(similarities[index]),6)} for index in order[:3]],
            'policy':{'minimum_similarity':manifest['minimum_similarity'],
                      'minimum_margin':manifest['minimum_margin']}}


def rank_student(embedding, text, weights, bias, manifest):
    """A supervised head has separate scores and gating from text similarity."""
    vector=np.asarray(embedding,dtype=np.float32).reshape(-1)
    if vector.shape!=(manifest['embedding_size'],) or not np.isfinite(vector).all():
        raise ValueError('Invalid student embedding')
    norm=np.linalg.norm(vector)
    if norm<1e-8:raise ValueError('Empty student embedding')
    vector=vector/norm
    classifier=manifest['classifier']
    logits=(weights@vector+bias)/classifier['temperature']
    scores=np.exp(logits-logits.max())
    scores=scores/scores.sum()
    if not np.isfinite(scores).all():raise ValueError('Invalid student scores')
    order=np.argsort(-scores,kind='stable')
    best,second=map(int,order[:2])
    margin=float(scores[best]-scores[second])
    label=manifest['labels'][best]
    supported=manifest.get('training',{}).get('train_material_counts',{})
    accepted=(classifier['qualified_for_acceptance']
              and float(scores[best])>=classifier['minimum_probability']
              and margin>=classifier['minimum_margin']
              and label not in manifest.get('rejection_labels',[])
              and supported.get(label,0)>=3)
    similarities=text@vector
    return {'status':'identified' if accepted else 'uncertain','label':label if accepted else None,
            'best_candidate':label,'score':round(float(scores[best]),6),
            'similarity':round(float(similarities[best]),6),'margin':round(margin,6),
            'score_type':'supervised_material_head_score',
            'candidates':[{'label':manifest['labels'][int(index)],'score':round(float(scores[index]),6),
                           'similarity':round(float(similarities[index]),6)} for index in order[:3]],
            'policy':{key:classifier[key] for key in
                      ['minimum_probability','minimum_margin','temperature','qualified_for_acceptance']}}


class MaterialClassifier:
    def __init__(self,directory,device='CPU',threads=2):
        import openvino as ov
        directory=Path(directory)
        self.manifest=json.loads((directory/'manifest.json').read_text())
        classifier=self.manifest.get('classifier')
        expected={'vision.xml','vision.bin','text_embeddings.npy'}
        if classifier is not None:
            if (classifier.get('kind')!='linear_softmax' or classifier.get('file')!='classifier.npz'
                    or not .1<=classifier.get('temperature',0)<=10
                    or not 0<=classifier.get('minimum_probability',-1)<=1
                    or not 0<=classifier.get('minimum_margin',-1)<=1
                    or type(classifier.get('qualified_for_acceptance')) is not bool):
                raise ValueError('Invalid student classifier policy')
            expected.add('classifier.npz')
        for name,digest in self.manifest['files'].items():
            if name not in expected:
                raise ValueError('Unexpected material artifact')
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('SigLIP2 artifact checksum mismatch')
        if set(self.manifest['files'])!=expected:
            raise ValueError('Missing material artifact checksums')
        self.text=np.load(directory/'text_embeddings.npy',allow_pickle=False)
        labels=self.manifest['labels']
        if (self.text.shape!=(len(labels),self.manifest['embedding_size']) or len(labels)<2
                or len(set(labels))!=len(labels) or not np.isfinite(self.text).all()
                or not np.allclose(np.linalg.norm(self.text,axis=1),1,atol=1e-4)):
            raise ValueError('Invalid material vocabulary embeddings')
        if not all(isinstance(label,str) and 0<len(label)<=80 for label in labels):
            raise ValueError('Invalid material labels')
        if not (0<=self.manifest['minimum_similarity']<=1 and 0<=self.manifest['minimum_margin']<=1):
            raise ValueError('Invalid material acceptance policy')
        self.student=None
        if classifier is not None:
            with np.load(directory/'classifier.npz',allow_pickle=False) as head:
                weights,bias=head['weight'].copy(),head['bias'].copy()
            if (weights.shape!=(len(labels),self.manifest['embedding_size']) or bias.shape!=(len(labels),)
                    or weights.dtype!=np.float32 or bias.dtype!=np.float32
                    or not np.isfinite(weights).all() or not np.isfinite(bias).all()):
                raise ValueError('Invalid student classifier weights')
            self.student=(weights,bias)
        # Validate the exact preprocessing contract before compiling the model.
        prepare(np.zeros((16,16,3),np.uint8),self.manifest)
        options={'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':1,'INFERENCE_PRECISION_HINT':'f32'}
        if device=='CPU':options['INFERENCE_NUM_THREADS']=threads
        self.compiled=ov.Core().compile_model(str(directory/'vision.xml'),device,options)
        if list(self.compiled.input(0).shape)!=[1,3,self.manifest['input_size'],self.manifest['input_size']]:
            raise ValueError('SigLIP2 input contract mismatch')
        self.output=self.compiled.output(0)
        self.config={'model':self.manifest['model'],'revision':self.manifest['revision'],
                     'manifest_sha256':hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
                     'artifact_sha256':self.manifest['files']['vision.bin'],
                     'runtime':'openvino','runtime_version':ov.__version__,'device':device,
                     'input_size':self.manifest['input_size'],'threads':threads,
                     'model_config':{key:self.manifest[key] for key in
                         ['processor','labels','prompts','rejection_labels','files']}}
        if classifier is not None:
            self.config.update(classifier=classifier,head_sha256=self.manifest['files']['classifier.npz'],
                               training=self.manifest.get('training',{}))

    def classify(self,rgb):
        tensor=prepare(rgb,self.manifest)
        started=time.monotonic()
        embedding=self.compiled([tensor])[self.output]
        result=(rank_student(embedding,self.text,*self.student,self.manifest) if self.student is not None
                else rank(embedding,self.text,self.manifest))
        return {**result,**self.config,'inference_ms':round((time.monotonic()-started)*1000,2)}
