"""Four-hop results must be compared with answers that need no four-hop computation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'neural'))
from train_liquid_recurrence import graph_cases,hop_baselines


class RecurrenceBaselineTests(unittest.TestCase):
    def test_validation_baselines_match_recorded_protocol(self):
        train=graph_cases(1024,1107)
        validation=graph_cases(64,1113,[tuple(row['graph']) for row in train])
        result=hop_baselines(validation)
        self.assertEqual(result['total'],512)
        # The start node coincides with the four-hop target far more often than chance.
        self.assertEqual((result['start'],result['hop1']),(167,137))
        self.assertEqual(result['best'],167)

    def test_single_eight_cycle_has_no_shortcut(self):
        graph=[1,2,3,4,5,6,7,0]
        target=[graph[graph[graph[graph[i]]]] for i in range(8)]
        result=hop_baselines([dict(graph=graph,target=target)])
        self.assertEqual(result['best'],0)


if __name__=='__main__':unittest.main()
