"""Boarding gallery and immediate top-1 alighting retrieval.

Only boarding creates identities. Alighting searches currently onboard people
by appearance, even if its stop-local track ID was seen before. No competition,
provisional assignments or deferred finalization is used.
"""
import numpy as np

from transitbox.tracks import normalize_activity


class IdentityGallery:
    def __init__(self, threshold=None):
        if threshold is not None and not -1 <= threshold <= 1:
            raise ValueError('Cosine threshold must be between -1 and 1')
        self.threshold = threshold
        self.identities = {}
        self.onboard = set()
        self.track_events = {}
        self.counter = 0

    def size(self, source):
        return sum(self.identities[pid]['source'] == source for pid in self.onboard)

    def associate(self, task, feature):
        source, stop = task['source'], task['stop']
        activity = normalize_activity(task.get('activity'))
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(feature)
        if not np.isfinite(feature).all() or norm < 1e-8:
            raise ValueError('Invalid ReID feature')
        feature = feature / norm
        keys = [(source, stop, tid) for tid in task['trackIds']]
        before = self.size(source)
        result = {
            'reid': None, 'reidStatus': 'awaiting_activity',
            'reidSimilarity': None, 'matchedClip': None, 'candidates': [],
            'reidBoardingClip': None, 'reidBoardingStop': None,
            'reidPayment': None, 'reidPaymentConfidence': None,
            'galleryBefore': before, 'galleryAfter': before,
            'galleryAction': 'none', 'activity': activity,
        }
        if activity == 'inside':
            result['reidStatus'] = 'skipped_inside'
            return result
        if activity not in ('boarding', 'alighting'):
            return result

        # Only reuse the latest event if its activity agrees. A board/exit/board
        # sequence creates a new boarding, even when the local tracker reuses an ID.
        previous = next((self.track_events[k] for k in keys
                         if k in self.track_events and self.track_events[k]['activity'] == activity), None)
        if previous is not None:
            result.update(previous)
            result.update({
                'reidStatus': 'boarding_repeat' if activity == 'boarding' else 'alighting_repeat',
                'galleryAction': 'none', 'galleryBefore': before, 'galleryAfter': before,
                'repeatedClip': previous['eventClip'],
            })
            for key in keys:
                self.track_events[key] = previous
            return result

        if activity == 'boarding':
            self.counter += 1
            identity = f'P{self.counter:04d}'
            passenger = {
                'source': source, 'feature': feature, 'boardEnd': task['end'],
                'boardStop': stop, 'boardClip': task['id'],
                'payment': task.get('payment'),
                'paymentConfidence': task.get('paymentConfidence'),
            }
            self.identities[identity] = passenger
            self.onboard.add(identity)
            result.update({'reid': identity, 'reidStatus': 'boarded', 'galleryAction': 'added'})
        else:
            candidates = []
            for pid in sorted(self.onboard):
                passenger = self.identities[pid]
                # Completed boarding only; earlier boarding at this stop is eligible.
                if passenger['source'] != source or passenger['boardEnd'] > task['start']:
                    continue
                similarity = float(np.clip(np.dot(feature, passenger['feature']), -1, 1))
                candidates.append({'identity': pid, 'similarity': similarity,
                                   'clip': passenger['boardClip'], 'stop': passenger['boardStop']})
            candidates.sort(key=lambda candidate: candidate['similarity'], reverse=True)
            result['candidates'] = [dict(candidate, similarity=round(candidate['similarity'], 5))
                                    for candidate in candidates[:3]]
            result['reidStatus'] = 'unmatched'
            result['unmatchedReason'] = 'empty_gallery' if not candidates else 'below_threshold'
            if candidates and (self.threshold is None or candidates[0]['similarity'] >= self.threshold):
                best = candidates[0]
                identity = best['identity']
                passenger = self.identities[identity]
                self.onboard.remove(identity)
                result.update({'reid': identity, 'reidStatus': 'matched',
                               'reidSimilarity': round(best['similarity'], 5), 'matchedClip': best['clip'],
                               'galleryAction': 'removed', 'unmatchedReason': None})

        if result['reid'] is not None:
            passenger = self.identities[result['reid']]
            result.update({'reidBoardingClip': passenger['boardClip'],
                           'reidBoardingStop': passenger['boardStop'],
                           'reidPayment': passenger['payment'],
                           'reidPaymentConfidence': passenger['paymentConfidence']})
        result['galleryAfter'] = self.size(source)
        result['eventClip'] = task['id']
        for key in keys:
            self.track_events[key] = result.copy()
        return result
