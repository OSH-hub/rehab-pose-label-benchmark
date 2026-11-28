"""Locked Penn Action common-12 full-frame evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import time
import numpy as np

DEFAULT_PATHS = r'rehabal_exp\paths.json'
PENN12 = np.arange(1, 13, dtype=np.int64)
JOINT_NAMES = ['left_shoulder','right_shoulder','left_elbow','right_elbow',
               'left_wrist','right_wrist','left_hip','right_hip',
               'left_knee','right_knee','left_ankle','right_ankle']
MAPS = {'rehab26': [7,12,8,13,9,14,16,21,17,22,18,23],
        'coco17': [5,6,7,8,9,10,11,12,13,14,15,16]}
EXPECTED = {'test_clips':1068, 'test_frames':76298, 'train_clips':1258, 'train_frames':87543}
PREP_CONFIG = {'schema_version':2, 'split_field':'train', 'test_value':-1,
               'penn_joint_indices_0based':PENN12.tolist(), 'gt_origin':'one_based',
               'internal_origin':'zero_based', 'bbox':'xyxy', 'bbox_scale':'max(width,height)',
               'visibility':'exactly_1_and_finite_xy', 'invalid_bbox':'exclude_from_PCK_denominator_only',
               'out_of_image_visible_gt':'retain_without_clipping',
               'missing_trailing_bbox_rows':'pad_nan_without_interpolation', 'expected':EXPECTED}
INFERENCE_CONFIG = {'imgsz':640, 'rect':False, 'conf':0.25, 'iou':0.70, 'max_det':300,
                    'augment':False, 'half':False, 'classes':[0],
                    'person_selection':'highest_box_confidence_first_on_tie',
                    'gt_crop':False, 'keypoint_confidence_filter':False}
THRESHOLDS = {'020':0.2, '010':0.1}


def now(): return datetime.now(timezone.utc).isoformat()

def lp(path):
    path=os.path.abspath(os.fspath(path))
    if os.name=='nt' and not path.startswith('\\\\?\\'):
        path='\\\\?\\UNC\\'+path[2:] if path.startswith('\\\\') else '\\\\?\\'+path
    return Path(path)

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(4*1024*1024)
            if not b: break
            h.update(b)
    return h.hexdigest()

def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True)

def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()

def log(**values): print(json.dumps(values,ensure_ascii=True),flush=True)

def read_json(path):
    with open(path,'r',encoding='utf-8-sig') as f: return json.load(f)

def atomic_new_json(path,value):
    path=Path(path)
    if path.exists():
        old=read_json(path)
        if old==value: return
        raise RuntimeError('Refusing non-identical overwrite: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.writing-'+str(os.getpid()))
    try:
        with temp.open('x',encoding='utf-8',newline='\n') as f:
            json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
            f.flush();os.fsync(f.fileno())
        if path.exists():raise RuntimeError('Concurrent output creation: '+str(path))
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)

def save_npz_new(path,**arrays):
    path=Path(path)
    if path.exists():raise RuntimeError('Refusing NPZ overwrite: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.writing-'+str(os.getpid()))
    try:
        with temp.open('xb') as f:
            np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        if path.exists():raise RuntimeError('Concurrent output creation: '+str(path))
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)

def to_zero_based(xy): return np.asarray(xy,dtype=np.float64)-1.0

def frame_key(clip_id,frame_id): return f'{str(clip_id).zfill(4)}/{int(frame_id):06d}'

def gt_mask(gt_xy,visibility,bbox):
    gt_xy=np.asarray(gt_xy);visibility=np.asarray(visibility);bbox=np.asarray(bbox)
    width=bbox[:,2]-bbox[:,0];height=bbox[:,3]-bbox[:,1]
    valid_bbox=np.isfinite(bbox).all(axis=1)&(width>0)&(height>0)
    valid_joint=(visibility==1)&np.isfinite(gt_xy).all(axis=2)
    return valid_joint&valid_bbox[:,None],valid_bbox,np.maximum(width,height)

def choose_instance(box_confidence,keypoints):
    if box_confidence is None or keypoints is None:return None
    boxes=np.asarray(box_confidence).reshape(-1)
    kp=np.asarray(keypoints)
    if kp.ndim!=3 or kp.shape[0]!=len(boxes):raise ValueError('Prediction instance dimensions disagree.')
    if not len(boxes):return None
    valid=np.isfinite(boxes)
    if not valid.any():return None
    index=int(np.argmax(np.where(valid,boxes,-np.inf)))
    return index,kp[index],float(boxes[index])

def score_predictions(gt_xy,visibility,bbox,pred_xy,detected):
    gt_xy=np.asarray(gt_xy,dtype=np.float64);pred_xy=np.asarray(pred_xy,dtype=np.float64)
    if gt_xy.shape!=pred_xy.shape or gt_xy.ndim!=3 or gt_xy.shape[1:]!=(12,2):
        raise ValueError('Expected matching [N,12,2] coordinate arrays.')
    detected=np.asarray(detected,dtype=bool)
    if detected.shape!=(len(gt_xy),):raise ValueError('Detection mask shape mismatch.')
    mask,valid_bbox,scale=gt_mask(gt_xy,visibility,bbox)
    finite=np.isfinite(pred_xy).all(axis=2)&detected[:,None]
    errors=np.full(mask.shape,np.inf,dtype=np.float64)
    eligible=mask&finite
    distance=np.linalg.norm(pred_xy-gt_xy,axis=2)
    np.divide(distance,scale[:,None],out=errors,where=eligible)
    correct={label:mask&(errors<=thr) for label,thr in THRESHOLDS.items()}
    return {'mask':mask,'bbox_valid':valid_bbox,'scale':scale,'errors':errors,'correct':correct}

def extract_mat_info(blob,name):
    from scipy.io import loadmat
    d=loadmat(io.BytesIO(blob))
    required=['x','y','visibility','bbox','dimensions','train','nframes','action']
    if any(k not in d for k in required):raise ValueError('Missing MAT field: '+name)
    n=int(d['nframes'].item());tr=int(d['train'].item())
    if tr not in (-1,1):raise ValueError('Unexpected official split: '+name)
    for k in ('x','y','visibility'):
        if np.asarray(d[k]).shape!=(n,13):raise ValueError('Invalid '+k+' shape: '+name)
    source_bbox=np.asarray(d['bbox'],dtype=np.float64)
    if source_bbox.ndim!=2 or source_bbox.shape[1]!=4 or source_bbox.shape[0]>n:
        raise ValueError('Invalid bbox shape: '+name)
    bbox=np.full((n,4),np.nan,dtype=np.float64)
    bbox[:len(source_bbox)]=source_bbox
    if not np.isin(d['visibility'],[0,1]).all():raise ValueError('Unknown visibility encoding.')
    dimensions=np.asarray(d['dimensions']).reshape(-1).astype(int)
    if dimensions.size!=3 or dimensions[2]!=n:raise ValueError('MAT dimension/frame mismatch.')
    return {'nframes':n,'train':tr,'action':str(d['action'].reshape(-1)[0]),
            'xy':to_zero_based(np.stack([d['x'][:,PENN12],d['y'][:,PENN12]],axis=2)),
            'visibility':np.asarray(d['visibility'][:,PENN12],dtype=np.uint8),
            'bbox':to_zero_based(bbox), 'source_bbox_rows':len(source_bbox),
            'missing_bbox_frame_ids':list(range(len(source_bbox)+1,n+1)),
            'height':int(dimensions[0]),'width':int(dimensions[1])}

def write_same_or_new_text(path,text):
    text=text.replace('\r\n','\n').replace('\r','\n')
    if path.exists():
        if path.read_text(encoding='utf-8')!=text:raise RuntimeError('Existing text differs: '+str(path))
    else:
        with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)

def load_paths(path):
    config=read_json(path)
    return {k:lp(v) for k,v in config.items()}

def prepare(paths,protocol):
    archive=paths['data_root']/'04_PennAction'/'Penn_Action.tar.gz'
    prepared=paths['output']/'penn_action'/'prepared'
    cache=paths['work']/'penn_cache'/'test_frames'
    prepared.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    protocol_hash=sha256_file(protocol)
    log(stage='prepare',action='hash_source_archive')
    archive_hash=sha256_file(archive)
    identity={'archive_sha256':archive_hash,'protocol_sha256':protocol_hash,'config':PREP_CONFIG}
    prepare_hash=digest(identity)
    final_meta=prepared/'prepare_complete.json'
    if final_meta.exists():
        old=read_json(final_meta)
        if old['prepare_sha256']!=prepare_hash:raise RuntimeError('Prepared cache config/source mismatch.')
        for name,key in [('gt_test.npz','gt_npz_sha256'),('manifest_test.jsonl','manifest_sha256')]:
            if sha256_file(prepared/name)!=old[key]:raise RuntimeError('Prepared artifact checksum mismatch: '+name)
        log(stage='prepare',status='already_complete',clips=old['test_clips'],frames=old['test_frames'])
        return old
    lock=prepared/'prepare.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    try:
        labels={};tools={};split_clips=Counter();split_frames=Counter()
        log(stage='prepare',action='read_official_annotations_only')
        with tarfile.open(archive,'r|gz') as tf:
            for mem in tf:
                if not mem.isfile():continue
                match=re.fullmatch(r'Penn_Action/labels/(\d{4})\.mat',mem.name)
                if match:
                    cid=match.group(1)
                    if cid in labels:raise ValueError('Duplicate MAT clip ID.')
                    if mem.size>32*1024*1024:raise ValueError('Unexpected MAT size.')
                    blob=tf.extractfile(mem).read()
                    info=extract_mat_info(blob,mem.name)
                    split_clips[info['train']]+=1;split_frames[info['train']]+=info['nframes']
                    labels[cid]=info
                elif mem.name in {'Penn_Action/tools/CreatePointLightDisplay.m','Penn_Action/tools/visualize.m'}:
                    tools[Path(mem.name).name]=tf.extractfile(mem).read().decode('utf-8')
        if dict(split_clips)!={-1:1068,1:1258} or dict(split_frames)!={-1:76298,1:87543}:
            raise ValueError('Official split counts differ from locked release: '+str((split_clips,split_frames)))
        if 'sub2ind' not in tools.get('CreatePointLightDisplay.m',''):
            raise ValueError('Official one-based coordinate evidence unavailable.')
        for name,text in tools.items():write_same_or_new_text(prepared/('official_'+name),text)
        test_ids=sorted(cid for cid,v in labels.items() if v['train']==-1)
        train_ids=sorted(cid for cid,v in labels.items() if v['train']==1)
        rows=[];xys=[];visibility=[];boxes=[];clip_index=[];frame_ids=[];widths=[];heights=[]
        for ci,cid in enumerate(test_ids):
            d=labels[cid];n=d['nframes']
            xys.append(d['xy']);visibility.append(d['visibility']);boxes.append(d['bbox'])
            clip_index.extend([ci]*n);frame_ids.extend(range(1,n+1));widths.extend([d['width']]*n);heights.extend([d['height']]*n)
            for fi in range(1,n+1):
                rows.append({'index':len(rows),'frame_key':frame_key(cid,fi),'clip_id':cid,'frame_id':fi,
                             'action':d['action'],'height':d['height'],'width':d['width'],
                             'image_path':str(cache/cid/f'{fi:06d}.jpg'),'official_train_flag':-1})
        gt=np.concatenate(xys).astype(np.float64);vis=np.concatenate(visibility);bbox=np.concatenate(boxes).astype(np.float64)
        eligible,bbox_valid,scale=gt_mask(gt,vis,bbox)
        out_of_bounds=(vis==1)&np.isfinite(gt).all(axis=2)&((gt[:,:,0]<0)|(gt[:,:,0]>=np.asarray(widths)[:,None])|(gt[:,:,1]<0)|(gt[:,:,1]>=np.asarray(heights)[:,None]))
        lookup={r['frame_key']:r for r in rows};found=set()
        log(stage='prepare',action='copy_only_official_test_jpegs',frames=len(rows))
        with tarfile.open(archive,'r|gz') as tf:
            for mem in tf:
                match=re.fullmatch(r'Penn_Action/frames/(\d{4})/(\d{6})\.jpg',mem.name)
                if not match or not mem.isfile():continue
                key=frame_key(match.group(1),int(match.group(2)))
                if key not in lookup:continue
                if key in found:raise ValueError('Duplicate test frame member: '+key)
                if mem.size>32*1024*1024:raise ValueError('Unexpected image size.')
                blob=tf.extractfile(mem).read()
                if len(blob)!=mem.size:raise ValueError('Truncated source JPEG.')
                checksum=hashlib.sha256(blob).hexdigest();row=lookup[key];target=Path(row['image_path'])
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if target.stat().st_size!=len(blob) or sha256_file(target)!=checksum:
                        raise RuntimeError('Existing extracted JPEG differs; refusing overwrite: '+key)
                else:
                    part=target.with_name(target.name+'.partial')
                    if part.exists():
                        prior=part.read_bytes()
                        if prior!=blob:raise RuntimeError('Interrupted JPEG differs; remove only this derived partial after review: '+str(part))
                    else:
                        with part.open('xb') as f:f.write(blob)
                    os.replace(part,target)
                row['image_bytes']=len(blob);row['image_sha256']=checksum;found.add(key)
                if len(found)%10000==0:log(stage='prepare',extracted_test_frames=len(found))
        if found!=set(lookup):raise ValueError('Missing expected official test JPEGs.')
        arrays={'gt_xy':gt,'visibility':vis,'bbox_xyxy':bbox,'bbox_valid':bbox_valid,
                'bbox_scale':scale,'evaluable_mask':eligible,'clip_index':np.asarray(clip_index,dtype=np.int32),
                'frame_ids':np.asarray(frame_ids,dtype=np.int32),'clip_ids':np.asarray(test_ids,dtype='U4'),
                'clip_actions':np.asarray([labels[c]['action'] for c in test_ids]),
                'frame_keys':np.asarray([r['frame_key'] for r in rows]),
                'prepare_sha256':np.asarray(prepare_hash),'protocol_sha256':np.asarray(protocol_hash)}
        npz=prepared/'gt_test.npz'
        if npz.exists():
            with np.load(npz,allow_pickle=False) as z:
                if str(z['prepare_sha256'])!=prepare_hash:raise RuntimeError('Existing GT NPZ identity mismatch.')
                for k,v in arrays.items():
                    if not np.array_equal(z[k],v,equal_nan=True) if np.issubdtype(np.asarray(v).dtype,np.number) else not np.array_equal(z[k],v):
                        raise RuntimeError('Existing GT NPZ content differs: '+k)
        else:save_npz_new(npz,**arrays)
        manifest=prepared/'manifest_test.jsonl'
        text=''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in rows)
        write_same_or_new_text(manifest,text)
        metadata={'status':'complete','created_at_utc':now(),'prepare_sha256':prepare_hash,
                  **identity,'archive_path':str(archive),'cache_path':str(cache),'test_clips':len(test_ids),'test_frames':len(rows),
                  'train_clip_ids':train_ids,'test_clip_ids':test_ids,'gt_npz_sha256':sha256_file(npz),
                  'manifest_sha256':sha256_file(manifest),'joint_names':JOINT_NAMES,
                  'gt_quality':{'invalid_bbox_frames':int((~bbox_valid).sum()),
                   'test_missing_bbox_frames':sum(len(labels[c]['missing_bbox_frame_ids']) for c in test_ids),
                   'missing_bbox_rows_by_clip':{c:{'train':v['train'],'nframes':v['nframes'],
                       'source_bbox_rows':v['source_bbox_rows'],'missing_frame_ids':v['missing_bbox_frame_ids']}
                       for c,v in labels.items() if v['missing_bbox_frame_ids']},
                   'visible_common12_joints':int((vis==1).sum()),'evaluable_joints':int(eligible.sum()),
                   'nonfinite_visible_gt':int(((vis==1)&~np.isfinite(gt).all(axis=2)).sum()),
                   'visible_common12_outside_image':int(out_of_bounds.sum()),
                   'zero_evaluable_clips':[c for i,c in enumerate(test_ids) if not eligible[np.asarray(clip_index)==i].any()]},
                  'official_matlab_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in tools.items()},
                  'prediction_performed':False}
        atomic_new_json(final_meta,metadata)
        log(stage='prepare',status='complete',clips=len(test_ids),frames=len(rows),gt_quality=metadata['gt_quality'],prediction_performed=False)
        return metadata
    finally:lock.unlink(missing_ok=True)

def load_prepared(paths,protocol):
    folder=paths['output']/'penn_action'/'prepared'
    meta=read_json(folder/'prepare_complete.json')
    if meta['status']!='complete' or sha256_file(protocol)!=meta['protocol_sha256']:
        raise RuntimeError('Prepared dataset is incomplete or protocol changed.')
    for name,key in [('gt_test.npz','gt_npz_sha256'),('manifest_test.jsonl','manifest_sha256')]:
        if sha256_file(folder/name)!=meta[key]:raise RuntimeError('Prepared checksum mismatch.')
    with np.load(folder/'gt_test.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with (folder/'manifest_test.jsonl').open('r',encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    if len(rows)!=EXPECTED['test_frames']:raise ValueError('Manifest frame count mismatch.')
    if len(set(r['frame_key'] for r in rows))!=len(rows):raise ValueError('Manifest duplicate frame keys.')
    return meta,data,rows

def infer_model_shape(model):
    head=model.model.model[-1]
    shape=getattr(head,'kpt_shape',None)
    if shape is None:shape=getattr(model.model,'yaml',{}).get('kpt_shape')
    if shape is None:raise ValueError('Cannot verify checkpoint pose head shape.')
    return [int(v) for v in shape]

def infer_chunk(model,rows,joint_map,shape,device,batch):
    for row in rows:
        if sha256_file(row['image_path'])!=row['image_sha256']:
            raise RuntimeError('Image cache checksum mismatch: '+row['frame_key'])
    result=[]
    for start in range(0,len(rows),batch):
        current=rows[start:start+batch]
        part=model.predict(source=[r['image_path'] for r in current],device=device,batch=batch,
                           verbose=False,stream=False,**INFERENCE_CONFIG_FOR_API)
        if len(part)!=len(current):raise ValueError('Predictor dropped an input frame.')
        result.extend(part)
    xy=np.full((len(rows),12,2),np.nan,dtype=np.float32)
    kconf=np.full((len(rows),12),np.nan,dtype=np.float32)
    detected=np.zeros(len(rows),dtype=bool);bconf=np.full(len(rows),np.nan,dtype=np.float32)
    for i,res in enumerate(result):
        rp=Path(str(res.path))
        if frame_key(rp.parent.name,int(rp.stem))!=rows[i]['frame_key']:
            raise ValueError('Predictor output order/clip key mismatch.')
        if res.boxes is None or res.keypoints is None:continue
        boxes=res.boxes.conf.detach().cpu().numpy()
        kp=res.keypoints.data.detach().cpu().numpy()
        choice=choose_instance(boxes,kp)
        if choice is None:continue
        _,instance,score=choice
        if instance.shape!=(shape[0],shape[1]):raise ValueError('Prediction head shape differs from checkpoint.')
        xy[i]=instance[joint_map,:2];kconf[i]=instance[joint_map,2]
        detected[i]=True;bconf[i]=score
    return xy,kconf,detected,bconf

INFERENCE_CONFIG_FOR_API={k:v for k,v in INFERENCE_CONFIG.items() if k in {'imgsz','rect','conf','iou','max_det','augment','half','classes'}}

def aggregate(model_entry,checkpoint_hash,protocol_hash,gt,pred_xy,detected):
    scored=score_predictions(gt['gt_xy'],gt['visibility'],gt['bbox_xyxy'],pred_xy,detected)
    rows=[]
    for i,cid in enumerate(gt['clip_ids']):
        use=gt['clip_index']==i;den=int(scored['mask'][use].sum())
        row={'model_id':model_entry['model_id'],'method':model_entry['method'],'seed':model_entry['seed'],
             'budget':model_entry['budget'],'backbone':model_entry['backbone'],'checkpoint_sha256':checkpoint_hash,
             'protocol_sha256':protocol_hash,'clip_id':str(cid),'action':str(gt['clip_actions'][i]),
             'n_frames':int(use.sum()),'n_frames_with_detection':int(detected[use].sum()),
             'n_visible_joints':int((gt['visibility'][use]==1).sum()),'n_evaluable_joints':den,
             'n_invalid_bbox_frames':int((~scored['bbox_valid'][use]).sum())}
        for tag in THRESHOLDS:
            count=int(scored['correct'][tag][use].sum())
            row['correct_'+tag]=count;row['pck_'+tag]=(count/den if den else None)
        rows.append(row)
    metrics={}
    denominator=int(scored['mask'].sum())
    for tag in THRESHOLDS:
        values=[r['pck_'+tag] for r in rows if r['pck_'+tag] is not None]
        metrics['clip_macro_pck_'+tag]=float(np.mean(values)) if values else None
        metrics['micro_pck_'+tag]=int(scored['correct'][tag].sum())/denominator if denominator else None
    metrics.update({'detection_rate':float(np.mean(detected)),'evaluable_joints':denominator,
                    'zero_evaluable_clips':sum(r['n_evaluable_joints']==0 for r in rows),
                    'clip_count':len(rows),'frame_count':len(detected),
                    'invalid_bbox_frames':int((~scored['bbox_valid']).sum())})
    return rows,metrics

def predict(paths,protocol,registry_path,device,batch,chunk_size):
    from ultralytics import YOLO
    import ultralytics,torch
    from ultralytics.engine.results import Keypoints
    probe=torch.tensor([[[23.5,17.25,0.01]]])
    if not torch.equal(Keypoints(probe.clone(),(100,100)).data,probe):
        raise RuntimeError('Installed Keypoints changes low-confidence coordinates; incompatible with frozen protocol.')
    meta,gt,manifest=load_prepared(paths,protocol)
    registry=read_json(registry_path)
    models=registry.get('models')
    if not isinstance(models,list) or not models:raise ValueError('Registry must contain nonempty models list.')
    ids=[m.get('model_id') for m in models]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate registry model IDs.')
    versions={'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'ultralytics':ultralytics.__version__}
    import cv2, PIL
    versions.update({'opencv':cv2.__version__,'pillow':PIL.__version__})
    code_hash=sha256_file(Path(__file__))
    for entry in models:
        required=['model_id','checkpoint','method','seed','budget','backbone','skeleton','expected_kpt_shape']
        if any(k not in entry for k in required):raise ValueError('Incomplete registry entry.')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}',entry['model_id']):raise ValueError('Unsafe model ID.')
        skeleton=entry['skeleton']
        if skeleton not in MAPS:raise ValueError('Unsupported skeleton mapping.')
        expected_shape=[26,3] if skeleton=='rehab26' else [17,3]
        if list(entry['expected_kpt_shape'])!=expected_shape:raise ValueError('Registry skeleton/shape mismatch.')
        checkpoint=lp(entry['checkpoint']);checkpoint_hash=sha256_file(checkpoint)
        if entry.get('checkpoint_sha256',checkpoint_hash)!=checkpoint_hash:raise ValueError('Checkpoint SHA256 mismatch.')
        identity={'schema_version':1,'model_entry':entry,'checkpoint_sha256':checkpoint_hash,
                  'protocol_sha256':meta['protocol_sha256'],'prepare_sha256':meta['prepare_sha256'],
                  'manifest_sha256':meta['manifest_sha256'],'gt_npz_sha256':meta['gt_npz_sha256'],
                  'evaluator_sha256':code_hash,'inference':INFERENCE_CONFIG,'joint_map':MAPS[skeleton],
                  'device':str(device),'batch':batch,'chunk_size':chunk_size,'package_versions':versions}
        fingerprint=digest(identity)
        folder=paths['output']/'penn_action'/'evaluations'/entry['model_id'];folder.mkdir(parents=True,exist_ok=True)
        identity_path=folder/'evaluation_identity.json'
        atomic_new_json(identity_path,{'fingerprint':fingerprint,**identity})
        complete=folder/'metrics.json'
        if complete.exists():
            old=read_json(complete)
            if old.get('status')!='complete' or old.get('fingerprint')!=fingerprint:raise RuntimeError('Existing metrics identity mismatch.')
            for filename,key in [('predictions.npz','predictions_sha256'),('per_clip_raw.csv','per_clip_raw_sha256')]:
                if sha256_file(folder/filename)!=old[key]:raise RuntimeError('Existing result checksum mismatch.')
            log(model_id=entry['model_id'],status='already_complete');continue
        lock=folder/'prediction.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
        try:
            model=YOLO(str(checkpoint));shape=infer_model_shape(model)
            if shape!=expected_shape:raise ValueError('Actual checkpoint kpt_shape mismatch: '+str(shape))
            n=len(manifest);xy=np.full((n,12,2),np.nan,dtype=np.float32);kc=np.full((n,12),np.nan,dtype=np.float32)
            found=np.zeros(n,dtype=bool);bc=np.full(n,np.nan,dtype=np.float32)
            chunks=folder/'prediction_chunks';chunks.mkdir(exist_ok=True)
            for start in range(0,n,chunk_size):
                stop=min(start+chunk_size,n);cp=chunks/f'{start:06d}_{stop:06d}.npz'
                expected_keys=np.asarray([r['frame_key'] for r in manifest[start:stop]])
                if cp.exists():
                    with np.load(cp,allow_pickle=False) as z:
                        if str(z['fingerprint'])!=fingerprint or not np.array_equal(z['frame_keys'],expected_keys):
                            raise RuntimeError('Resume chunk identity mismatch: '+str(cp))
                        cx=z['xy'].copy();ck=z['keypoint_confidence'].copy();cd=z['detected'].copy();cb=z['box_confidence'].copy()
                else:
                    cx,ck,cd,cb=infer_chunk(model,manifest[start:stop],MAPS[skeleton],shape,device,batch)
                    save_npz_new(cp,fingerprint=np.asarray(fingerprint),frame_keys=expected_keys,xy=cx,
                                 keypoint_confidence=ck,detected=cd,box_confidence=cb)
                if cx.shape!=(stop-start,12,2) or ck.shape!=(stop-start,12) or cd.shape!=(stop-start,) or cb.shape!=(stop-start,):
                    raise ValueError('Malformed prediction chunk.')
                xy[start:stop]=cx;kc[start:stop]=ck;found[start:stop]=cd;bc[start:stop]=cb
                log(stage='inference',model_id=entry['model_id'],frames_processed=stop,total_frames=n)
            predfile=folder/'predictions.npz'
            if predfile.exists():
                with np.load(predfile,allow_pickle=False) as z:
                    if str(z['fingerprint'])!=fingerprint:raise RuntimeError('Existing full predictions identity mismatch.')
                    if not np.array_equal(z['xy'],xy,equal_nan=True) or not np.array_equal(z['detected'],found):
                        raise RuntimeError('Existing full predictions differ from exact chunks.')
            else:save_npz_new(predfile,xy=xy,keypoint_confidence=kc,detected=found,box_confidence=bc,
                              frame_keys=gt['frame_keys'],fingerprint=np.asarray(fingerprint),checkpoint_sha256=np.asarray(checkpoint_hash))
            clip_rows,stats=aggregate(entry,checkpoint_hash,meta['protocol_sha256'],gt,xy,found)
            buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=list(clip_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(clip_rows)
            csvpath=folder/'per_clip_raw.csv';write_same_or_new_text(csvpath,buffer.getvalue())
            result={'status':'complete','completed_at_utc':now(),'fingerprint':fingerprint,
                    'model_id':entry['model_id'],'method':entry['method'],'seed':entry['seed'],'budget':entry['budget'],
                    'backbone':entry['backbone'],'checkpoint_sha256':checkpoint_hash,'protocol_sha256':meta['protocol_sha256'],
                    'prepare_sha256':meta['prepare_sha256'],'actual_kpt_shape':shape,'joint_map':MAPS[skeleton],
                    'official_test_clips_expected':1068,'official_test_clips_processed':len(clip_rows),
                    'official_test_frames_expected':76298,'official_test_frames_processed':n,
                    'primary':'clip_macro_pck_020','secondary':['clip_macro_pck_010','micro_pck_020','micro_pck_010'],
                    'metrics':stats,'gt_quality':meta['gt_quality'],'predictions_sha256':sha256_file(predfile),
                    'per_clip_raw_sha256':sha256_file(csvpath),'inference_config':INFERENCE_CONFIG,'package_versions':versions}
            atomic_new_json(complete,result)
            log(stage='evaluation',model_id=entry['model_id'],status='complete',metrics_path=str(complete))
            del model
            if torch.cuda.is_available():torch.cuda.empty_cache()
        finally:lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',default=DEFAULT_PATHS)
    actions=p.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare',action='store_true');actions.add_argument('--predict',action='store_true')
    p.add_argument('--registry');p.add_argument('--device',default='0');p.add_argument('--batch',type=int,default=32)
    p.add_argument('--chunk-size',type=int,default=256)
    args=p.parse_args();paths=load_paths(args.paths);protocol=paths['output']/'protocol_penn.txt'
    if args.batch<1 or args.chunk_size<1:raise ValueError('Batch/chunk sizes must be positive.')
    if args.prepare:prepare(paths,protocol)
    else:
        if not args.registry:p.error('--predict requires --registry')
        predict(paths,protocol,args.registry,args.device,args.batch,args.chunk_size)
    return 0

if __name__=='__main__':raise SystemExit(main())
