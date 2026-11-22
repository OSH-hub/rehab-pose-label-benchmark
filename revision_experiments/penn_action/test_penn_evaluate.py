"""Synthetic protocol checks; no model inference or real test-set scores."""
import inspect
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import penn_evaluate as pe


class ProtocolTests(unittest.TestCase):
    def make_gt(self,n=2):
        return (np.full((n,12,2),50.0),np.ones((n,12),dtype=np.uint8),np.tile([0.,0.,100.,100.],(n,1)))

    def test_missing_detection_is_failure_in_denominator(self):
        gt,vis,bbox=self.make_gt()
        out=pe.score_predictions(gt,vis,bbox,gt.copy(),[True,False])
        self.assertEqual(int(out['mask'].sum()),24)
        self.assertEqual(int(out['correct']['020'].sum()),12)

    def test_clip_frame_composite_key(self):
        self.assertEqual(pe.frame_key('1',1),'0001/000001')
        self.assertNotEqual(pe.frame_key('0001',1),pe.frame_key('0002',1))

    def test_one_based_to_zero_based_once(self):
        np.testing.assert_array_equal(pe.to_zero_based([[1.,1.],[15.5,20.25]]),[[0.,0.],[14.5,19.25]])

    def test_person_choice_uses_box_score_without_gt(self):
        kp=np.stack([np.zeros((26,3)),np.full((26,3),999.)])
        choice=pe.choose_instance([.3,.9],kp)
        self.assertEqual(choice[0],1)
        self.assertEqual(set(inspect.signature(pe.choose_instance).parameters),{'box_confidence','keypoints'})

    def test_person_tie_preserves_first_output(self):
        choice=pe.choose_instance([.8,.8],np.zeros((2,26,3)))
        self.assertEqual(choice[0],0)

    def test_visible_mask_does_not_depend_on_prediction_confidence(self):
        gt,vis,bbox=self.make_gt(1);vis[0,0]=0
        out=pe.score_predictions(gt,vis,bbox,gt,[True])
        self.assertEqual(int(out['mask'].sum()),11)
        self.assertEqual(int(out['correct']['020'].sum()),11)

    def test_nonfinite_prediction_fails_without_dropping_denominator(self):
        gt,vis,bbox=self.make_gt(1);pred=gt.copy();pred[0,0,0]=np.nan
        out=pe.score_predictions(gt,vis,bbox,pred,[True])
        self.assertEqual(int(out['mask'].sum()),12)
        self.assertEqual(int(out['correct']['020'].sum()),11)

    def test_bbox_max_side_threshold_and_invalid_bbox(self):
        gt,vis,bbox=self.make_gt(2);bbox[0]=[0.,0.,200.,100.];bbox[1]=[0.,0.,0.,100.]
        pred=gt.copy();pred[:,:,0]+=40.
        out=pe.score_predictions(gt,vis,bbox,pred,[True,True])
        self.assertEqual(int(out['mask'].sum()),12)
        self.assertEqual(int(out['correct']['020'].sum()),12)
        self.assertEqual(int(out['correct']['010'].sum()),0)

    def test_outside_image_visible_joint_is_not_clipped(self):
        gt,vis,bbox=self.make_gt(1);gt[0,0]=[-10.,200.]
        out=pe.score_predictions(gt,vis,bbox,gt.copy(),[True])
        self.assertEqual(int(out['mask'].sum()),12)
        self.assertEqual(int(out['correct']['020'].sum()),12)

    def test_mapping_excludes_head_and_proximal_clavicle(self):
        self.assertEqual(pe.PENN12.tolist(),list(range(1,13)))
        self.assertEqual(pe.MAPS['rehab26'][:2],[7,12])
        self.assertNotIn(6,pe.MAPS['rehab26']);self.assertNotIn(11,pe.MAPS['rehab26'])
        self.assertEqual(len(set(pe.MAPS['rehab26'])),12)

    def test_clip_macro_and_micro_are_separate(self):
        gt,vis,bbox=self.make_gt(3)
        data={'gt_xy':gt,'visibility':vis,'bbox_xyxy':bbox,'clip_index':np.array([0,1,1]),
              'clip_ids':np.array(['0001','0002']),'clip_actions':np.array(['a','b'])}
        model={'model_id':'synthetic','method':'test','seed':0,'budget':.4,'backbone':'none'}
        rows,metrics=pe.aggregate(model,'x','y',data,gt.copy(),np.array([True,False,False]))
        self.assertEqual(metrics['clip_macro_pck_020'],.5)
        self.assertAlmostEqual(metrics['micro_pck_020'],1/3)
        self.assertEqual(rows[1]['n_evaluable_joints'],24)
        self.assertEqual(rows[1]['correct_020'],0)

    def test_zero_evaluable_clip_is_null_not_zero(self):
        gt,vis,bbox=self.make_gt(2);vis[1]=0
        data={'gt_xy':gt,'visibility':vis,'bbox_xyxy':bbox,'clip_index':np.array([0,1]),
              'clip_ids':np.array(['0001','0002']),'clip_actions':np.array(['a','b'])}
        model={'model_id':'synthetic','method':'test','seed':0,'budget':.4,'backbone':'none'}
        rows,metrics=pe.aggregate(model,'x','y',data,gt.copy(),np.array([True,False]))
        self.assertIsNone(rows[1]['pck_020']);self.assertEqual(metrics['clip_macro_pck_020'],1.)

    def test_explicit_source_batches_bound_memory_and_keep_all_frames(self):
        calls=[]
        class FakeModel:
            def predict(self,source,**kwargs):
                assert kwargs['rect'] is False and kwargs['imgsz']==640
                calls.append(len(source))
                return [SimpleNamespace(path=p,boxes=None,keypoints=None) for p in source]
        rows=[{'image_path':str(Path('0001')/f'{i:06d}.jpg'),
               'image_sha256':'x','frame_key':pe.frame_key('0001',i)} for i in range(1,6)]
        with patch.object(pe,'sha256_file',return_value='x'):
            xy,kc,detected,bc=pe.infer_chunk(FakeModel(),rows,pe.MAPS['rehab26'],[26,3],'cpu',2)
        self.assertEqual(calls,[2,2,1])
        self.assertEqual(xy.shape,(5,12,2))
        self.assertFalse(detected.any())

    def test_missing_trailing_bbox_is_not_interpolated(self):
        from scipy.io import savemat
        buffer=io.BytesIO()
        savemat(buffer,{'x':np.ones((2,13)),'y':np.ones((2,13)),
                        'visibility':np.ones((2,13)), 'bbox':np.array([[1.,2.,11.,12.]]),
                        'dimensions':np.array([100,100,2]), 'train':-1,'nframes':2,'action':'test'})
        info=pe.extract_mat_info(buffer.getvalue(),'synthetic.mat')
        np.testing.assert_array_equal(info['bbox'][0],[0.,1.,10.,11.])
        self.assertTrue(np.isnan(info['bbox'][1]).all())
        self.assertEqual(info['missing_bbox_frame_ids'],[2])
        self.assertEqual(info['source_bbox_rows'],1)
        mask,valid,_=pe.gt_mask(info['xy'],info['visibility'],info['bbox'])
        self.assertEqual(valid.tolist(),[True,False])
        self.assertEqual(int(mask.sum()),12)
    def test_exact_identity_only_can_resume_json(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'identity.json'
            pe.atomic_new_json(p,{'checkpoint':'a','conf':.25})
            pe.atomic_new_json(p,{'checkpoint':'a','conf':.25})
            with self.assertRaises(RuntimeError):pe.atomic_new_json(p,{'checkpoint':'b','conf':.25})
            self.assertEqual(pe.read_json(p)['checkpoint'],'a')


if __name__=='__main__':unittest.main()
