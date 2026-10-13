"""Source discrimination cannot pass on ordinary or duplicate conditions."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'neural'))
from liquid_stream_training import source_control_summary


def record(word=None,identity='case',missing=False):
    metadata={} if word is None else dict(selfReportIntervention=word,sourceControlId=identity)
    expected=[dict(kind=kind,exact=True,forcedFeedback=False) for kind in
              (['fruit'] if word is None else ['self-report','external-report'])]
    if missing:expected=expected[:1]
    return dict(metadata=metadata,expected=expected)


class SourceEvaluationTests(unittest.TestCase):
    def test_unrelated_conversations_and_order_do_not_create_pairs(self):
        result=source_control_summary([record(),record('Mango.'),record(),record('Apple.')])
        self.assertEqual(result['sourceControlPairs'],1)
        self.assertEqual(result['sourceControlPairsCorrect'],1)
        self.assertNotIn('fruit',result['roleScores'])

    def test_duplicate_or_incomplete_conditions_do_not_pass(self):
        for rows in [[record('Apple.'),record('Apple.')],
                     [record('Apple.',missing=True),record('Mango.')],
                     [record('Apple.')]]:
            result=source_control_summary(rows)
            self.assertEqual(result['sourceControlPairsCorrect'],0)
            self.assertEqual(result['invalidSourceControlGroups'],1)


def target(answer,at,kind='self-report'):
    return dict(kind=kind,answer=answer,questionSeconds=at,forcedFeedback=False)


def said(text,at,ended=True,intervened=False):
    return dict(text=text,startSeconds=at,ended=ended,intervened=intervened)


class StrictUtteranceScoringTests(unittest.TestCase):
    def test_emitting_both_alternatives_fails(self):
        from liquid_stream_training import match_targets
        matched,false_starts=match_targets([target('Apple.',10.)],[said('Apple.',10.03),said('Mango.',10.24)])
        self.assertFalse(matched[0]['exact']);self.assertEqual(matched[0]['utterancesInWindow'],2)
        matched,_=match_targets([target('Apple.',10.)],[said('Mango.',10.03),said('Apple.',10.24)])
        self.assertFalse(matched[0]['exact'])

    def test_single_correct_answer_passes_and_stray_output_is_counted(self):
        from liquid_stream_training import match_targets
        matched,false_starts=match_targets([target('Apple.',10.)],[said('Hello.',4.),said('Apple.',10.06)])
        self.assertTrue(matched[0]['exact']);self.assertEqual(false_starts,1)
        self.assertAlmostEqual(matched[0]['latencyMilliseconds'],60.,places=3)

    def test_unfinished_or_injected_utterances_do_not_count(self):
        from liquid_stream_training import match_targets
        matched,false_starts=match_targets([target('Apple.',10.)],[said('Mango.',2.,intervened=True),said('Apple.',10.1,ended=False)])
        self.assertFalse(matched[0]['exact']);self.assertEqual(false_starts,0)


if __name__=='__main__':unittest.main()
