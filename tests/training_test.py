import sys
import tempfile
import subprocess,json
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'world_model'))
from model import MLP
from train import target_scale
from model import read_json
from compare_runs import seed_metrics, interval
from structured import StructuredWorld
from train_value import targets


class TrainingTests(unittest.TestCase):
    def test_only_present_players_contribute_value_examples_and_opponent_targets(self):
        from train_value import active_samples
        schema=json.loads(subprocess.check_output(['node','-e',"console.log(JSON.stringify(require('./world_model/encoding').schema))"],cwd=Path(__file__).resolve().parents[1]))
        ix={f['name']:i for i,f in enumerate(schema['fields'])}
        states=np.zeros((3,len(ix)),'float32');states[:,ix['players']]=[.5,.75,1]
        labels=np.array([[100,80,9999,9999],[60,70,90,9999],[10,20,30,40]],'float32')
        x,y=active_samples(states,labels,schema,.25,'brass-value-v2')
        self.assertEqual(x.shape,(9,125));self.assertEqual(y.shape,(9,1))
        np.testing.assert_allclose(y[:,0]*100,[80,55,37.5,47.5,72.5,0,10,20,32.5])
        labels[0,2:]=-9999;labels[1,3]=-9999
        x2,y2=active_samples(states,labels,schema,.25,'brass-value-v2')
        np.testing.assert_array_equal(x,x2);np.testing.assert_array_equal(y,y2)

    def test_dynamics_replay_uses_training_rows_and_restores_row_order(self):
        from dataset import TrainingMixture
        class FakeDataset:
            schema={'version':'test-v1','fields':['a','b']};state_dim=2;action_dim=1;input_dim=3
            def __init__(self,seed,offset):
                self.games=[dict(seed=seed)];self.splits={'train':[1,3,5,7]};self.offset=offset
            def batch(self,ids):
                ids=np.asarray(ids);v=(ids+self.offset).astype('float32')[:,None]
                return np.repeat(v,3,1),np.repeat(-v,2,1),v
        primary=FakeDataset(1,0);replay=FakeDataset(2,100)
        mixed=TrainingMixture(primary,replay,2,27)
        self.assertEqual(mixed.size,6);self.assertTrue(set(mixed.rows[1])<={1,3,5,7})
        order=np.array([5,0,4,3,1,2]);x,d,y=mixed.batch(order)
        expected=np.concatenate([mixed.rows[0],mixed.rows[1]+100])[order]
        np.testing.assert_array_equal(x[:,0],expected);np.testing.assert_array_equal(d[:,0],-expected)
        np.testing.assert_array_equal(y[:,0],expected)
        np.testing.assert_array_equal(mixed.rows[1],TrainingMixture(primary,replay,2,27).rows[1])
        with self.assertRaisesRegex(ValueError,'overlap'):TrainingMixture(primary,FakeDataset(1,100))

    def test_geographic_features_match_javascript_over_game_phases(self):
        from value_model import features
        script="""const S=require('./world_model/simulator'),E=require('./world_model/encoding'),V=require('./world_model/value_features');
let s=S.create(4,82701),cases=[];for(let i=0;!s.gameOver;i++){if(i%9===0){const v=E.encodeState(s);cases.push({v,f:[0,1,2,3].map(p=>V.features(v,p,'brass-value-v2'))});}s=S.step(s,S.candidates(s).sort((a,b)=>b.score-a.score)[0]).state;}
console.log(JSON.stringify({schema:E.schema,cases}));"""
        data=json.loads(subprocess.check_output(['node','-e',script],cwd=Path(__file__).resolve().parents[1]))
        states=np.array([c['v'] for c in data['cases']],dtype='float32')
        result=features(states,data['schema'],'brass-value-v2')
        self.assertEqual(result.shape,(len(states),4,125))
        np.testing.assert_allclose(result,np.array([c['f'] for c in data['cases']]),atol=2e-6)
        np.testing.assert_array_equal(result[:,:,:109],features(states,data['schema']))

    def test_score_objective_uses_own_final_vp_and_keeps_legacy_margin(self):
        scores=np.array([[100,80,90,50],[20,30,40,50]],dtype='float32')
        np.testing.assert_allclose(targets(scores,0)*100,scores)
        changed=scores.copy();changed[:,1:]+=200
        np.testing.assert_array_equal(targets(scores,0)[:,0],targets(changed,0)[:,0])
        self.assertAlmostEqual(float(targets(scores,.25)[0,0]),.775,places=6)

    def test_replay_targets_exclude_final_income_only_when_requested(self):
        from train_value import replay_samples
        from model import write_json
        schema=json.loads(subprocess.check_output(['node','-e',"console.log(JSON.stringify(require('./world_model/encoding').schema))"],cwd=Path(__file__).resolve().parents[1]))
        ix={f['name']:i for i,f in enumerate(schema['fields'])};n=len(ix)
        rows=np.zeros((2,n+4),'float32');rows[:,ix['players']]=1;rows[:,-4:]=[100,80,70,60]
        rows[0,ix['p0.income']]=.9;rows[1,ix['p0.income']]=.5
        schema.update(stateDim=n,rowWidth=n+4,rows=2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);rows.tofile(path/'states.f32');write_json(path/'schema.json',schema)
            write_json(path/'games.json',[dict(seed=7,start=0,end=2,split='train')])
            x,raw,_=replay_samples(path,'train',schema,0)
            x2,adjusted,_=replay_samples(path,'train',schema,0,income_bonus_weight=1)
        np.testing.assert_array_equal(x,x2)
        np.testing.assert_allclose(raw.reshape(2,4)[:,0],[1,1])
        np.testing.assert_allclose(adjusted.reshape(2,4)[:,0],[.85,.85])

    def test_structured_resume_restores_all_heads_and_next_update(self):
        schema=json.loads(subprocess.check_output(['node','-e',"console.log(JSON.stringify(require('./world_model/encoding').schema))"],cwd=Path(__file__).resolve().parents[1]))
        schema.update(stateDim=len(schema['fields']),actionDim=147);dimension=schema['stateDim']
        model=StructuredWorld(schema,np.ones(dimension),np.full(dimension,.1),hidden=8)
        x=np.zeros((3,dimension+schema['actionDim']),'float32');d=x[:,:dimension].copy()
        model.update(x,d,.001);artifact=model.export()
        restored=StructuredWorld.load(artifact);restored.schema=schema
        with tempfile.TemporaryDirectory() as directory:
            for name in ['delta','gate','control']:
                a=getattr(model,name);b=getattr(restored,name);checkpoint=Path(directory)/(name+'.npz')
                a.save_optimizer(checkpoint);b.load_optimizer(checkpoint)
        model.update(x,d,.0005);restored.update(x,d,.0005)
        for name in ['delta','gate','control']:
            for a,b in zip(getattr(model,name).params,getattr(restored,name).params):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(model.predict(x),restored.predict(x))

    def test_restoring_optimizer_matches_uninterrupted_next_update(self):
        rng=np.random.default_rng(15)
        x=rng.normal(size=(16,8)).astype('float32');y=rng.normal(size=(16,3)).astype('float32')
        uninterrupted=MLP(8,3,5,11)
        uninterrupted.update(x,y,lr=.0003)
        restored=MLP.load(uninterrupted.export())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'optimizer.npz'
            uninterrupted.save_optimizer(checkpoint);restored.load_optimizer(checkpoint)
        uninterrupted.update(x,y,lr=.0002);restored.update(x,y,lr=.0002)
        for a,b in zip(uninterrupted.params,restored.params):np.testing.assert_array_equal(a,b)

    def test_residual_score_targets_match_browser_anchor_and_exclude_absent_seats(self):
        from value_model import score_anchors
        from train_value import active_samples
        script="""process.env.BRASS_RULES='economy-v2';const S=require('./world_model/simulator'),E=require('./world_model/encoding'),V=require('./world_model/value_features'),D=require('./js/gameData');
        const states=[];for(const n of [2,3,4]){const s=S.create(n,31);s.players[0].vp=12;s.boardIndustries.birmingham_0={playerId:0,type:'cottonMill',tileData:D.INDUSTRY_DATA.cottonMill[1],flipped:true,resourceCubes:0};s.boardLinks['birmingham-oxford']={playerId:0,type:'canal'};states.push(E.encodeState(s));s.gameOver=true;states.push(E.encodeState(s));}
        console.log(JSON.stringify({schema:E.schema,states,anchors:states.map(v=>[0,1,2,3].map(p=>V.scoreAnchor(v,p)))}));"""
        data=json.loads(subprocess.check_output(['node','-e',script],cwd=Path(__file__).resolve().parents[1]))
        states=np.array(data['states'],dtype='float32');anchors=score_anchors(states,data['schema'])
        np.testing.assert_allclose(anchors,data['anchors'],atol=1e-5)
        labels=np.full((6,4),100,dtype='float32')
        x,y=active_samples(states,labels,data['schema'],0,'brass-value-v2',True)
        self.assertEqual(len(x),18)
        counts=[2,2,3,3,4,4];mask=np.arange(4)[None,:]<np.array(counts)[:,None]
        np.testing.assert_allclose(y[:,0],((labels-anchors)/100)[mask],atol=1e-6)

    def test_chunked_scale_matches_dense_and_uses_only_selected_rows(self):
        rng=np.random.default_rng(33);values=rng.normal(size=(8500,4)).astype('float32')
        class FakeDataset:
            state_dim=4
            def batch(self,ids):return None,values[ids],None
        selected=np.arange(8200);values[8200:]=1000
        np.testing.assert_allclose(target_scale(FakeDataset(),selected),values[selected].std(0),rtol=2e-6)

    def test_seed_group_win_gaps_match_original_tournament_including_ties(self):
        report=read_json(Path(__file__).resolve().parents[1]/'world_model/experiments/v1/reports/tournament.json')
        seeds,metrics=seed_metrics(report)
        self.assertEqual(len(seeds),25)
        self.assertAlmostEqual(metrics[:,0].mean(),.035)
        self.assertAlmostEqual(metrics[:,1].mean(),.81)
        self.assertEqual(interval(metrics[:,0],np.tile(np.arange(25),(10,1)))['estimate'],metrics[:,0].mean())
        report['games'].pop()
        with self.assertRaisesRegex(ValueError,'four seat rotations'):seed_metrics(report)


if __name__=='__main__':unittest.main()
