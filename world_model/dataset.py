from pathlib import Path
import numpy as np
from model import read_json

class Dataset:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.schema = read_json(self.directory/'schema.json')
        self.splits = read_json(self.directory/'splits.json')
        self.games = read_json(self.directory/'games.json')
        s = self.schema
        self.state_dim, self.action_dim = s['stateDim'], s['actionDim']
        expected = s['rows'] * s['rowWidth'] * 4
        if (self.directory/'transitions.f32').stat().st_size != expected:
            raise ValueError('Incomplete or incompatible dataset')
        self.data = np.memmap(self.directory/'transitions.f32',mode='r',dtype='<f4',shape=(s['rows'],s['rowWidth']))
        self.input_dim = self.state_dim + self.action_dim

    def batch(self, indices):
        rows = np.asarray(self.data[indices])
        x = rows[:, :self.input_dim]
        delta = rows[:,self.input_dim:-1] - rows[:,:self.state_dim]
        return x, delta, rows[:,-1:]


class TrainingMixture:
    """Mix complete-game training splits without copying multi-GB datasets."""
    def __init__(self, primary, replay=None, replay_stride=8, seed=42):
        if replay_stride < 1:raise ValueError('Replay stride must be positive')
        self.sources=[primary];self.rows=[np.asarray(primary.splits['train'],dtype='int64')]
        if replay is not None:
            if primary.schema['fields']!=replay.schema['fields'] or primary.action_dim!=replay.action_dim:
                raise ValueError('Replay schema mismatch')
            if {g['seed'] for g in primary.games}&{g['seed'] for g in replay.games}:
                raise ValueError('Replay seeds overlap primary games')
            ids=np.asarray(replay.splits['train'],dtype='int64')
            # Random sampling avoids repeatedly excluding era-ending actions.
            selected=np.random.default_rng(seed).choice(ids,max(1,len(ids)//replay_stride),replace=False)
            self.sources.append(replay);self.rows.append(np.sort(selected))
        self.ends=np.cumsum([len(ids) for ids in self.rows]);self.size=int(self.ends[-1])
        self.state_dim=primary.state_dim;self.input_dim=primary.input_dim

    def batch(self, indices):
        indices=np.asarray(indices);x=np.empty((len(indices),self.input_dim),'float32')
        delta=np.empty((len(indices),self.state_dim),'float32');score=np.empty((len(indices),1),'float32')
        start=0
        for ds,rows,end in zip(self.sources,self.rows,self.ends):
            mask=(indices>=start)&(indices<end)
            if mask.any():x[mask],delta[mask],score[mask]=ds.batch(rows[indices[mask]-start])
            start=end
        if np.any(indices<0) or np.any(indices>=self.size):raise IndexError('Training mixture index outside range')
        return x,delta,score
