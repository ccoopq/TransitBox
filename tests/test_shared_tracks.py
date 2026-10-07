import unittest
import numpy as np
from transitbox.tracks import load_tasks,crop_track
from transitbox.matching import IdentityGallery
from transitbox.live_bridge import task_from_stream
from transitbox.tracks import ROOT
import json

class SharedTrackTests(unittest.TestCase):
    @unittest.skipUnless((ROOT.parent/'GHR-VLM'/'outputs_C3_1'/'results.json').is_file(), 'Requires local private GHR-VLM data')
    def test_real_tracks_crop_without_redetection(self):
        tasks=load_tasks();self.assertEqual(len(tasks),415)
        self.assertEqual(len({t['id'] for t in tasks}),415)
        for task in [tasks[0],next(t for t in tasks if t['source']=='C3_3')]:
            crops,chosen=crop_track(task)
            self.assertGreater(len(crops),0)
            self.assertTrue(all(c.ndim==3 and c.shape[0]>8 and c.shape[1]>8 for c in crops))
            self.assertTrue(all(d['trackId'] in task['trackIds'] for d in chosen))
            self.assertTrue(all(task['start']-.025<=d['time']<task['end'] for d in chosen))

    @unittest.skipUnless((ROOT.parent/'GHR-VLM'/'outputs_C3_1'/'results.json').is_file(), 'Requires local private GHR-VLM data')
    def test_stream_callback_uses_the_same_track_list(self):
        root=ROOT.parent/'GHR-VLM'/'outputs_C3_1'
        record=json.loads((root/'results.json').read_text())[0]
        record['saved_clip_path']=str(root/'passenger_clips'/__import__('pathlib').Path(record['saved_clip_path']).name)
        split=next(e for e in json.loads((root/'flow_events.json').read_text()) if e.get('type')=='passenger_split' and e['stop_id']==record['stop_id'])
        task=task_from_stream(record,split,'C3_1')
        expected=load_tasks()[0]
        self.assertEqual(task['id'],expected['id'])
        self.assertEqual(task['frames'],expected['frames'])
        self.assertEqual(task['trackIds'],expected['trackIds'])

    def task(self,id,activity='boarding',stop=1,start=0,track=1,source='C3_1',**extra):
        return {'source':source,'stop':stop,'trackIds':[track],'start':start,'end':start+1,
                'id':id,'activity':activity,**extra}

    def test_boarding_repeat_does_not_add_second_person(self):
        gallery=IdentityGallery();feature=np.array([1.,0.])
        a=gallery.associate(self.task('a'),feature)
        b=gallery.associate(self.task('b',start=2),feature)
        self.assertEqual(a['reid'],b['reid'])
        self.assertEqual(b['reidStatus'],'boarding_repeat')
        self.assertEqual(gallery.size('C3_1'),1)

    def test_exit_uses_appearance_even_with_same_stop_and_track(self):
        gallery=IdentityGallery()
        a=gallery.associate(self.task('a'),np.array([1.,0.]))
        b=gallery.associate(self.task('b',track=2,start=2,payment='Cash',paymentConfidence='high'),np.array([0.,1.]))
        c=gallery.associate(self.task('c','alighting',start=4),np.array([0.,1.]))
        self.assertEqual(c['reidStatus'],'matched')
        self.assertEqual(c['reid'],b['reid'])  # track 1 does not force identity a
        self.assertEqual(c['matchedClip'],'b')
        self.assertEqual(c['reidPayment'],'Cash')
        self.assertEqual(c['reidBoardingStop'],1)
        self.assertNotIn(b['reid'],gallery.onboard)
        self.assertIn(a['reid'],gallery.onboard)
        repeat=gallery.associate(self.task('d','alighting',start=6),np.array([0.,1.]))
        self.assertEqual(repeat['reidStatus'],'alighting_repeat')
        self.assertEqual(repeat['reid'],b['reid'])
        self.assertEqual(gallery.size('C3_1'),1)
        reboard=gallery.associate(self.task('e',start=8),np.array([0.,1.]))
        self.assertEqual(reboard['reidStatus'],'boarded')
        self.assertNotIn(reboard['reid'],(a['reid'],b['reid']))

    def test_gallery_contains_only_boardings_and_removes_matches(self):
        gallery=IdentityGallery();feature=np.array([1.,0.])
        empty=gallery.associate(self.task('exit','leaving_bus'),feature)
        self.assertEqual(empty['reidStatus'],'unmatched');self.assertIsNone(empty['reid'])
        inside=gallery.associate(self.task('inside','staying_inside',start=2),feature)
        self.assertEqual(inside['reidStatus'],'skipped_inside');self.assertFalse(gallery.onboard)
        unknown=gallery.associate(self.task('unknown','unknown',start=4),feature)
        self.assertEqual(unknown['reidStatus'],'awaiting_activity');self.assertFalse(gallery.onboard)
        board=gallery.associate(self.task('board',start=6),feature)
        exit=gallery.associate(self.task('exit2','alighting',stop=2,start=8),feature)
        self.assertEqual(exit['reid'],board['reid'])  # one candidate is enough
        self.assertEqual(exit['galleryAfter'],0)
        later=gallery.associate(self.task('exit3','alighting',stop=3,start=10),feature)
        self.assertEqual(later['reidStatus'],'unmatched');self.assertFalse(later['candidates'])

    def test_sources_and_future_boardings_cannot_match(self):
        gallery=IdentityGallery();feature=np.array([1.,0.])
        gallery.associate(self.task('future',start=10),feature)
        before=gallery.associate(self.task('earlier','alighting',stop=2,start=4),feature)
        other=gallery.associate(self.task('other','alighting',source='C3_3',start=12),feature)
        self.assertEqual(before['reidStatus'],'unmatched');self.assertFalse(before['candidates'])
        self.assertEqual(other['reidStatus'],'unmatched');self.assertEqual(gallery.size('C3_1'),1)

    def test_optional_threshold_leaves_rejected_identity_onboard(self):
        gallery=IdentityGallery(threshold=.9)
        gallery.associate(self.task('a'),np.array([1.,0.]))
        exit=gallery.associate(self.task('b','alighting',start=2),np.array([0.,1.]))
        self.assertEqual(exit['reidStatus'],'unmatched')
        self.assertEqual(exit['unmatchedReason'],'below_threshold')
        self.assertEqual(gallery.size('C3_1'),1)

    def test_live_late_activity_labels_reuse_features_and_preserve_time_order(self):
        from tools.reid_worker import replay_gallery
        early=self.task('early','unknown',start=0,clipId=1)
        exit=self.task('exit','alighting',stop=2,start=2,clipId=2)
        future=self.task('future',stop=3,start=10,clipId=3)
        # Deliberately deliver features out of order, as asynchronous workers may.
        tasks={t['id']:t for t in (future,exit,early)}
        features={key:np.array([1.,0.]) for key in tasks}
        metadata={key:{'error':None} for key in tasks}
        before,_=replay_gallery(tasks,features,metadata)
        self.assertEqual(before['early']['reidStatus'],'awaiting_activity')
        self.assertEqual(before['exit']['reidStatus'],'unmatched')
        early.update(activity='boarding',payment='Cash')
        after,onboard=replay_gallery(tasks,features,metadata)
        self.assertEqual(after['exit']['matchedClip'],'early')
        self.assertEqual(after['exit']['reidPayment'],'Cash')
        self.assertEqual(onboard['C3_1'],1)

if __name__=='__main__':unittest.main()
