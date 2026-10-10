import sys,json,tempfile,unittest,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'world_model'))
from train_human_intent import load_rows
from train_critical_controls import prepare,control_metrics
from structured import StructuredWorld
from model import read_json

ROOT=Path(__file__).resolve().parents[1]
class HumanTrainingTests(unittest.TestCase):
    def corpus(self):
        reports=[];rows=[]
        for i,split in enumerate(['train','validation','test']):
            reports.append(dict(sha256=str(i),source='source'+str(i),terminal=True,nativeLegalityAudited=True,split=split,humanSeats=[0],humanActions=1,actions=2,seed=i))
            rows.append(dict(gameSHA256=str(i),source='source'+str(i),split=split,actor=0,seq=0,seed=i,features=[0.],intents=['loan'],valueLabelUsableForJS=False,provenance='public-anonymized-human-vs-AI-replay'))
        return dict(reports=reports,humanIntentRows=3,featureNames=['x'],classes=['loan']),rows
    def load(self,audit,rows):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'audit.json').write_text(json.dumps(audit));(p/'human-intents.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            return load_rows(p)
    def test_complete_whole_game_human_corpus(self):
        audit,rows=self.corpus();self.assertEqual(len(self.load(audit,rows)[1]),3)
    def test_reject_ai_foreign_value_labels_and_tampered_split(self):
        for field,value in [('actor',1),('valueLabelUsableForJS',True),('split','validation'),('features',[float('nan')]),('source','wrong')]:
            audit,rows=self.corpus();rows[0][field]=value
            with self.assertRaises(ValueError):self.load(audit,rows)
    def test_reject_missing_and_duplicate_decisions(self):
        audit,rows=self.corpus()
        with self.assertRaises(ValueError):self.load(audit,rows[:-1])
        audit['humanIntentRows']=4
        with self.assertRaises(ValueError):self.load(audit,rows+[rows[0]])
    def test_critical_control_training_preserves_board_and_value(self):
        artifact=read_json(ROOT/'world_model/experiments/human-teacher-20261009/models/world-model.json')
        schema=read_json(ROOT/'world_model/experiments/human-teacher-20261009/reproducibility/augmented-resource-v2/schema.json')
        old=StructuredWorld.load(artifact);new=prepare(artifact,schema,41)
        np.testing.assert_array_equal(old.delta.params[0],new.delta.params[0]);np.testing.assert_array_equal(old.gate.params[0],new.gate.params[0])
        self.assertIn(new.state_dim+new.action_dim-1,new.spec['inputIndices'])
        self.assertEqual(new.spec['version'],'complete-action-control-v2')

if __name__=='__main__':unittest.main()
