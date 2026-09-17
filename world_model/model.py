"""Small, dependency-light NumPy MLP with Adam and portable JSON weights."""
import json
import numpy as np

class MLP:
    def __init__(self, input_dim, output_dim, hidden=96, seed=1):
        rng = np.random.default_rng(seed)
        self.params = [
            (rng.standard_normal((input_dim, hidden)) * np.sqrt(2 / input_dim)).astype('float32'),
            np.zeros(hidden, 'float32'),
            (rng.standard_normal((hidden, output_dim)) * 0.001).astype('float32'),
            np.zeros(output_dim, 'float32'),
        ]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.steps = 0

    def predict(self, x):
        w1,b1,w2,b2 = self.params
        return np.maximum(0, x @ w1 + b1) @ w2 + b2

    def update(self, x, y, weights=None, lr=0.001):
        w1,b1,w2,b2 = self.params
        h = np.maximum(0, x @ w1 + b1)
        err = h @ w2 + b2 - y
        weighted = err if weights is None else err * weights
        grad = 2 * weighted / err.size
        dh = (grad @ w2.T) * (h > 0)
        grads = [x.T @ dh, dh.sum(0), h.T @ grad, grad.sum(0)]
        self.apply_gradients(grads,lr)
        return float((err * weighted).mean())

    def update_gradient(self, x, grad, lr=0.001):
        w1,b1,w2,b2 = self.params
        h = np.maximum(0,x @ w1+b1)
        dh = (grad @ w2.T)*(h>0)
        self.apply_gradients([x.T @ dh,dh.sum(0),h.T @ grad,grad.sum(0)],lr)

    def apply_gradients(self, grads, lr):
        self.steps += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            np.clip(g, -2, 2, out=g)
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            p -= lr * (self.m[i] / (1 - 0.9 ** self.steps)) / (np.sqrt(self.v[i] / (1 - 0.999 ** self.steps)) + 1e-8)

    def export(self, **metadata):
        return dict(metadata, layers=[dict(weights=self.params[0].tolist(),bias=self.params[1].tolist(),activation='relu'),
                                     dict(weights=self.params[2].tolist(),bias=self.params[3].tolist(),activation='linear')])

    @classmethod
    def load(cls, data):
        a,b = data['layers']
        model = cls(len(a['weights']),len(b['bias']),len(a['bias']))
        model.params = [np.array(a['weights'],'float32'), np.array(a['bias'],'float32'),
                        np.array(b['weights'],'float32'), np.array(b['bias'],'float32')]
        return model

    def save_optimizer(self, path):
        np.savez(path, steps=self.steps, **{f'm{i}': x for i,x in enumerate(self.m)},
                 **{f'v{i}': x for i,x in enumerate(self.v)})

    def load_optimizer(self, path):
        with np.load(path, allow_pickle=False) as data:
            for i,p in enumerate(self.params):
                if data[f'm{i}'].shape != p.shape or data[f'v{i}'].shape != p.shape:
                    raise ValueError('Optimizer/model shape mismatch')
            self.steps = int(data['steps'])
            self.m = [data[f'm{i}'].copy() for i in range(4)]
            self.v = [data[f'v{i}'].copy() for i in range(4)]

def read_json(path):
    with open(path, encoding='utf-8') as f: return json.load(f)

def write_json(path, value):
    with open(path, 'w', encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,allow_nan=False)
