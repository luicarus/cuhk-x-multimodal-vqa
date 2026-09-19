import csv
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import yaml

from test_cached_inputs import project, write_csv, PROJECT, CACHE
from cuhkx.config import InputError, load_config
from cuhkx.data.inputs import QA_FIELDS
from cuhkx.training.dataset import load_training_config, prepare_data, require_ready, TrainingDataset
from cuhkx.training.collator import answer_labels, CompletionCollator
from cuhkx.training.adapter import text_targets, save_adapter_receipt, verify_adapter
from cuhkx.training.trainer import latest_checkpoint, checkpoint_receipt, train


@pytest.fixture
def training(project):
    shutil.copy2(PROJECT/'configs/training.yaml', project/'configs/training.yaml')
    config = load_config(project)
    config['baseline']['model']['revision'] = 'a'*40
    settings = load_training_config(config)
    data = project/'data'
    originals = data/f'frames/test/{CACHE}/test/clip'
    original_meta = json.loads((originals/'metadata.json').read_text())
    qa, folds, indices = [], [], {i:[] for i in range(5)}
    for i,fold in enumerate([0,1,2,3,4,4]):
        qid, clip = f't{i}', f'clip{i}'
        qa.append(dict(zip(QA_FIELDS,[qid,'HAU',clip,'single','Action?','A text','B text','C text','D text']),answer='A'))
        folds.append(dict(fold_version='subject_grouped_v1',qa_id=qid,clip_key='train:'+clip,
                          subject_id=f'subject{fold}',source='HAU',category='single',fold=fold))
        root = data/settings['caches'][fold]['frames_root']
        folder = root/CACHE/'train'/clip
        folder.mkdir(parents=True)
        for image in originals.glob('*.jpg'):shutil.copy2(image,folder/image.name)
        meta = {**original_meta,'qa_ids':[qid],'qa_count':1,'sample_id':clip,'clip_key':'train:'+clip,'split':'train'}
        (folder/'metadata.json').write_text(json.dumps(meta))
        row = {k:meta[k] for k in ('sample_id','clip_key','source','split','status','modality','config_hash','protocol_version','qa_count')}
        row.update(qa_ids=qid,metadata_path=f'{CACHE}/train/{clip}/metadata.json',
                   frame_paths=';'.join(f'{CACHE}/train/{clip}/{name}' for name in meta['frame_files']))
        for key in ('source_frame_indices','normalized_positions','target_timestamps_seconds','actual_timestamps_seconds'):
            row[key]=';'.join(map(str,meta[key]))
        indices[fold].append(row)
    for fold, entries in indices.items():
        write_csv(data/settings['caches'][fold]['frame_index'],list(entries[0]),entries)
    write_csv(data/'references/training_qa.csv',list(qa[0]),qa)
    write_csv(data/'references/folds/subject_grouped_v1/qa_folds.csv',list(folds[0]),folds)
    write_csv(data/'qa/pilot.csv',QA_FIELDS,[{k:qa[-1][k] for k in QA_FIELDS}])
    return config,settings


def test_split_coverage_and_answer_separation(training):
    config,settings=training
    prepared=prepare_data(config,settings)
    assert prepared['status']=='PASS'
    assert prepared['cache_summary']=={'linked_qa':6,'clips':6,'images_decoded':48}
    assert {k:c['available_qa'] for k,c in prepared['coverage'].items()}=={'train':3,'dev':1,'confirm':1}
    assert [s['qa']['qa_id'] for s in prepared['samples']['confirm']]==['t4']
    sample=TrainingDataset(prepared['samples']['train'],config['data_root'])[0]
    assert 'answer' not in sample['qa'] and sample['answer']=='A'
    assert [p.name for p in sample['frame_paths']]==[f'frame_{i:04}.jpg' for i in (1,3,5,7)]


def test_missing_cache_report_and_train_fails_before_model(training,tmp_path):
    config,settings=training
    (Path(config['data_root'])/settings['caches'][0]['frame_index']).unlink()
    result=prepare_data(config,settings)
    assert result['status']=='INCOMPLETE' and result['coverage']['train']['missing_qa']==1
    with pytest.raises(InputError,match='missing'):
        train(config,settings,tmp_path/'nonexistent-weights','unit')
    assert not (Path(config['project_root'])/'artifacts/training').exists()


def test_subject_leakage_rejected(training):
    config,settings=training
    path=Path(config['data_root'])/settings['folds']
    with path.open() as h:rows=list(csv.DictReader(h))
    rows[3]['subject_id']=rows[0]['subject_id']
    write_csv(path,list(rows[0]),rows)
    with pytest.raises(InputError,match='leaks'):
        prepare_data(config,settings)


def test_wrong_qa_cache_join_rejected(training):
    config,settings=training
    path=Path(config['data_root'])/settings['folds']
    with path.open() as h:rows=list(csv.DictReader(h))
    rows[0]['clip_key']='train:wrong_clip'
    write_csv(path,list(rows[0]),rows)
    with pytest.raises(InputError,match='clip'):
        prepare_data(config,settings)


def test_mask_preserves_real_eos_when_pad_has_same_id():
    labels,trailing=answer_labels([9,8,65,0,198,0],[1,1,1,1,1,0],[0,9,8],[0,1,1],[65],0,8)
    assert labels==[-100,-100,65,0,-100,-100] and trailing==[198]


@pytest.mark.parametrize('prefix,answer,maximum', [([9,7],[65],9),([9,8],[66],9),([9,8],[65],2)])
def test_bad_mask_boundary_answer_and_truncation(prefix,answer,maximum):
    with pytest.raises(InputError):
        answer_labels([9,8,65,0],[1,1,1,1],prefix,[1,1],answer,0,maximum)


def test_collator_uses_four_images_and_answer_only_mask(training,monkeypatch):
    config,settings=training
    samples=TrainingDataset(prepare_data(config,settings)['samples']['train'],config['data_root'])
    class Row(list):
        def tolist(self):return list(self)
    class Tokenizer:
        eos_token_id=0
        def encode(self,text,**kwargs):return [65]
        def decode(self,tokens,**kwargs):return '\n' if tokens==[198] else ''
    class Processor:
        tokenizer=Tokenizer()
        def apply_chat_template(self,messages,**kwargs):
            assert len(messages[0]['content'])==5
            assert all(x['resized_height']==280 for x in messages[0]['content'][:4])
            return 'full' if len(messages)==2 else 'prefix'
        def __call__(self,**kwargs):
            assert kwargs['truncation'] is False and len(kwargs['images'])==4
            full=kwargs['text']==['full']
            ids=[9,8,65,0,198] if full else [9,8]
            return {'input_ids':[Row(ids)],'attention_mask':[Row([1]*len(ids))],'pixel_values':'mock'}
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(tensor=lambda data,**kwargs:data,long='long'))
    monkeypatch.setitem(sys.modules,'qwen_vl_utils',SimpleNamespace(process_vision_info=lambda messages:([x['image'] for x in messages[0]['content'][:4]],None)))
    batch=CompletionCollator(Processor())([samples[0]])
    assert batch['labels']==[[-100,-100,65,0,-100]]


def test_adapter_targets_exclude_visual_and_other_modules():
    names=['model.language_model.layers.0.self_attn.q_proj','model.language_model.layers.0.self_attn.v_proj',
           'model.visual.layers.0.self_attn.q_proj','vision.layers.0.self_attn.v_proj','lm_head']
    assert text_targets(names)==names[:2]
    with pytest.raises(InputError):text_targets(names[2:])
    with pytest.raises(InputError,match='both'):
        text_targets(['model.layers.0.self_attn.q_proj','model.layers.1.self_attn.v_proj'])


def test_adapter_receipt_and_tamper(tmp_path):
    source={'model_id':'Qwen/Qwen2.5-VL-7B-Instruct','revision':'a'*40,'receipt_sha256':'base'}
    targets=['model.language_model.layers.0.self_attn.q_proj','model.language_model.layers.0.self_attn.v_proj']
    lora={'r':8,'alpha':16,'dropout':0.05}
    cfg={'peft_type':'LORA','task_type':'CAUSAL_LM','bias':'none','target_modules':targets,'r':8,'lora_alpha':16,'lora_dropout':0.05}
    (tmp_path/'adapter_config.json').write_text(json.dumps(cfg))
    (tmp_path/'adapter_model.safetensors').write_bytes(b'test-only')
    save_adapter_receipt(tmp_path,source,'training','data',lora,targets,'sft')
    assert verify_adapter(tmp_path,source)['purpose']=='sft'
    with pytest.raises(InputError,match='base model'):
        verify_adapter(tmp_path,{**source,'revision':'b'*40})
    (tmp_path/'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(InputError,match='hash'):
        verify_adapter(tmp_path,source)


def test_adapter_receipt_accepts_peft_compacted_targets(tmp_path):
    source={'model_id':'Qwen/Qwen2.5-VL-7B-Instruct','revision':'a'*40,'receipt_sha256':'base'}
    targets=['model.language_model.layers.0.self_attn.q_proj','model.language_model.layers.0.self_attn.v_proj']
    lora={'r':8,'alpha':16,'dropout':0.05}
    cfg={'peft_type':'LORA','task_type':'CAUSAL_LM','bias':'none','target_modules':targets,'r':8,'lora_alpha':16,'lora_dropout':0.05}
    (tmp_path/'adapter_config.json').write_text(json.dumps(cfg))
    (tmp_path/'adapter_model.safetensors').write_bytes(b'test-only')
    save_adapter_receipt(tmp_path,source,'training','data',lora,targets,'sft')
    compact=json.loads((tmp_path/'adapter_config.json').read_text())
    compact['target_modules']=['q_proj','v_proj']
    (tmp_path/'adapter_config.json').write_text(json.dumps(compact))
    save_adapter_receipt(tmp_path,source,'training','data',lora,targets,'sft')
    assert verify_adapter(tmp_path,source)['purpose']=='sft'


def test_training_checkpoint_completeness_and_hashes(tmp_path):
    complete=tmp_path/'checkpoint-1';complete.mkdir()
    for name in ('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth','scaler.pt'):
        (complete/name).write_bytes(b'test-only')
    checkpoint_receipt(complete,'sig')
    (tmp_path/'checkpoint-2').mkdir()
    assert latest_checkpoint(tmp_path,'sig')==complete
    receipt=json.loads((complete/'checkpoint_receipt.json').read_text())
    partial={**receipt,'files':{k:v for k,v in receipt['files'].items() if k!='rng_state.pth'}}
    (complete/'checkpoint_receipt.json').write_text(json.dumps(partial))
    with pytest.raises(InputError,match='recovery state'):latest_checkpoint(tmp_path,'sig')
    (complete/'checkpoint_receipt.json').write_text(json.dumps(receipt))
    (complete/'optimizer.pt').write_bytes(b'tamper')
    with pytest.raises(InputError,match='content changed'):latest_checkpoint(tmp_path,'sig')


def test_dev_evaluation_uses_fixed_answer_free_inputs(training):
    from cuhkx.training.evaluate import evaluation_input
    from cuhkx.inference.runner import run_predictions, verify_completed_run
    from test_ir4_runner import FakeBackend
    config,settings=training
    prepared,samples=evaluation_input(config,settings,'dev')
    assert [q['qa_id'] for q in prepared[0]]==['t3']
    assert all('answer' not in q for q in prepared[0])
    source={'model_id':config['baseline']['model']['id'],'revision':'a'*40}
    result=run_predictions(config,'dev',None,'dev_unit',source,lambda:FakeBackend(['A']),
                           execution_mode='simulation',prepared_input=prepared)
    assert result['status']=='PASS'
    _,targets,_,_,_=verify_completed_run(config,'dev_unit',prepared_input=prepared)
    assert len(targets)==1


def test_training_profile_rejects_fold_mixing(training,tmp_path):
    config,settings=training
    path=Path(config['project_root'])/'configs/training.yaml'
    settings['splits']['train'].append(4)
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(InputError,match='splits'):
        load_training_config(config)


def test_registered_training_image_cannot_change_silently(training):
    from PIL import Image
    from test_cached_inputs import seal
    config,settings=training
    data=Path(config['data_root'])
    seal(data)
    root=data/settings['caches'][0]['frames_root']
    image_path=next(root.rglob('frame_0000.jpg'))
    Image.new('RGB',(448,448),(10,20,30)).save(image_path)
    with pytest.raises(InputError,match='registered training asset changed'):
        prepare_data(config,settings)
