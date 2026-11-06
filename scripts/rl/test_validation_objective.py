import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from validation_objective import training_objective_metrics,macro_objective_metrics


class ObjectiveTests(unittest.TestCase):
    def fixture(self,root):
        n=32;order=[1,0]+list(range(2,n))
        (root/'states.txt').write_text(''.join(f'STATE {p}\n' for p in order))
        sizes={p:150 if p==0 else 100 for p in range(n)}
        (root/'stream.266').write_bytes(b'x'*sum(sizes.values()))
        (root/'au.csv').write_text('output_step,poc,au_bytes,au_bits\n'+''.join(
            f'{i},{p},{sizes[p]},{8*sizes[p]}\n' for i,p in enumerate(range(n),1)))
        box={'x1':0,'y1':0,'x2':10,'y2':10,'confidence':.9,'class_id':2}
        for name in ('ref','rec'):
            rows=[dict(video_id='video',poc=p,detections=[] if name=='ref' and p==1 else [box]) for p in range(n)]
            (root/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return n,order,sizes

    def evaluate(self,root,n,gamma):
        return training_objective_metrics(root/'stream.266',root/'states.txt',root/'au.csv',
            root/'ref.jsonl',root/'rec.jsonl','video',n,80,.1,gamma,root/'trace.jsonl')

    def test_reward_discount_empty_reference_and_decision_order(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);n,order,sizes=self.fixture(root)
            reward=[];cumulative=0
            for step,poc in enumerate(order,1):
                cumulative+=8*sizes[poc]
                detection=-.25 if poc==1 else 1.
                reward.append(detection-.1*abs(math.log((cumulative*30/step)/80000)))
            result=self.evaluate(root,n,.9)
            self.assertAlmostEqual(result['mean_training_reward'],sum(reward)/n)
            self.assertAlmostEqual(result['discounted_return'],sum((.9**i)*r for i,r in enumerate(reward)))
            trace=[json.loads(s) for s in (root/'trace.jsonl').read_text().splitlines()]
            self.assertEqual([r['poc'] for r in trace],order)
            self.assertAlmostEqual(trace[0]['detection_term'],-.25)
            self.assertAlmostEqual(self.evaluate(root,n,0.)['discounted_return'],reward[0])
            self.assertAlmostEqual(self.evaluate(root,n,1.)['discounted_return'],sum(reward))

    def test_missing_decision_or_inconsistent_au_bits_rejected(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);n,_,_=self.fixture(root)
            (root/'states.txt').write_text('STATE 0\n'*n)
            with self.assertRaisesRegex(ValueError,'POCs'):self.evaluate(root,n,.99)
            self.fixture(root);(root/'stream.266').write_bytes(b'wrong size')
            with self.assertRaisesRegex(ValueError,'accounting'):self.evaluate(root,n,.99)

    def test_macro_returns_are_balanced_by_pair_not_clip_f1(self):
        pairs=[{'ppo':{'score':.9,'mean_training_reward':.4,'discounted_return':20.},
                'vanilla':{'score':.7,'mean_training_reward':.5,'discounted_return':25.}},
               {'ppo':{'score':.9,'mean_training_reward':.6,'discounted_return':40.},
                'vanilla':{'score':.7,'mean_training_reward':.7,'discounted_return':45.}}]
        result=macro_objective_metrics(pairs)
        self.assertEqual(result['macro_ppo_discounted_return'],30.)
        self.assertEqual(result['macro_vanilla_discounted_return'],35.)
        self.assertEqual(result['macro_ppo_mean_training_reward'],.5)


if __name__=='__main__':unittest.main()
