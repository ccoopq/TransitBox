import unittest
import numpy as np

from transitbox.privacy import expanded_box, head_box, redact, TemporalFaces, uncovered_heads


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

    def test_partial_face_clipped_to_frame_and_head_fallback(self):
        image = np.random.default_rng(9).integers(0,256,(80,100,3),dtype=np.uint8)
        face = (-12,-4,24,20)
        result,count = redact(image,[face],[head_box((.5,.1,.9,.95),100,80)])
        self.assertGreater(count,0)
        self.assertFalse(np.array_equal(result[:20,:20],image[:20,:20]))
        self.assertFalse(np.array_equal(result[10:30,55:85],image[10:30,55:85]))

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
        self.assertEqual(tracker.update(.2,[]),[new])
        self.assertEqual(tracker.update(1.2,[]),[])

    def test_head_fallback_only_skipped_when_face_is_centrally_covered(self):
        head=(20,10,80,60)
        self.assertEqual(uncovered_heads([(40,15,15,20)],[head]),[])
        self.assertEqual(uncovered_heads([], [head]),[head])


if __name__=='__main__':
    unittest.main()
