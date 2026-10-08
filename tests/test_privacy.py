import unittest
import numpy as np

from transitbox.privacy import expanded_box, redact, TemporalFaces, valid_face_box


class PrivacyTests(unittest.TestCase):
    def test_mask_covers_face_and_preserves_surroundings(self):
        image = np.random.default_rng(7).integers(0,256,(100,120,3),dtype=np.uint8)
        face = (40,30,20,24)
        result, count = redact(image,[face])
        x1,y1,x2,y2 = expanded_box(face,120,100)
        self.assertEqual(count,(x2-x1)*(y2-y1))
        self.assertTrue(np.array_equal(result[:y1],image[:y1]))
        self.assertTrue(np.array_equal(result[y2:],image[y2:]))
        self.assertFalse(np.array_equal(result[y1:y2,x1:x2],image[y1:y2,x1:x2]))
        self.assertGreater(image[y1:y2,x1:x2].std(),result[y1:y2,x1:x2].std()*3)
        self.assertFalse(np.shares_memory(result,image))

    def test_partial_face_clipped_to_frame_without_body_mask(self):
        image = np.random.default_rng(9).integers(0,256,(80,100,3),dtype=np.uint8)
        face = (-12,-4,24,20)
        result,count = redact(image,[face])
        self.assertGreater(count,0)
        self.assertFalse(np.array_equal(result[:20,:20],image[:20,:20]))
        self.assertTrue(np.array_equal(result[10:60,55:85],image[10:60,55:85]))

    def test_no_face_does_not_change_pixels(self):
        image=np.zeros((30,40,3),np.uint8)
        result,count=redact(image,[])
        self.assertEqual(count,0)
        self.assertTrue(np.array_equal(image,result))

    def test_temporal_recovery_holds_misses_without_trails(self):
        tracker=TemporalFaces()
        old=(30,20,12,18);new=(34,21,12,18)
        self.assertEqual(tracker.update(0,[old]),[old])
        self.assertEqual(tracker.update(.1,[new]),[new])
        self.assertTrue(np.allclose(tracker.update(.2,[]),[(38,22,12,18)]))
        self.assertEqual(tracker.update(1.2,[]),[])

    def test_large_weak_equipment_detection_is_rejected(self):
        self.assertFalse(valid_face_box((170,157,108,145),.42,640,360))
        self.assertTrue(valid_face_box((280,122,34,42),.81,640,360))
        self.assertTrue(valid_face_box((412,116,32,43),.32,640,360))
        self.assertTrue(valid_face_box((10,10,70,100),.6,100,160,scene_guard=False))

    def test_face_mask_is_local_and_does_not_cover_torso(self):
        image=np.random.default_rng(12).integers(0,256,(150,120,3),dtype=np.uint8)
        face=(40,15,20,24)
        result,count=redact(image,[face])
        self.assertLessEqual(count,20*24*2)
        self.assertTrue(np.array_equal(result[55:],image[55:]))

    def test_tracker_does_not_merge_different_face_sizes(self):
        tracker=TemporalFaces()
        small=(30,20,12,18);large=(20,10,80,120)
        tracker.update(0,[small])
        result=tracker.update(.1,[large])
        self.assertEqual(len(result),2)



if __name__=='__main__':
    unittest.main()
