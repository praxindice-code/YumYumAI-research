"""Espresso extraction benchmark: published kinetic curves and generic neural baselines on held-out brew settings.

Data: Angeloni et al. 2023, Foods 12:2871 (doi:10.3390/foods12152871, CC BY 4.0), supplementary
TableS1_ExperimentRawData.xlsx. 15 conditions (flow 1/2/3 mL/s x grind 1.4/1.7/2.0 x 80/89/98 C), 48 shots, fractions
1,2,3,5,7,10 analysed for trigonelline, caffeine, 5-CQA and TDS. Download it yourself; it is not redistributed here.

Arms (inputs: flow, grind, temperature, fraction masses; loss: MSE on log concentration; config fixed in advance):
- curve: the paper's model c = c0*exp(-m/tau); log c0 and log tau linear in the settings.
- curve2: two-stage (fast + slow) exponential; every parameter linear in the settings.
- mlp: settings + cumulative mass -> log concentrations.
- gru: GRU over percolation blocks -> log concentrations.
Splits: 'condition' (leave one condition out) and 'extrapolate' (hold out a whole extreme level of flow, grind or
temperature, so test settings lie outside the training range). Paired sign-flip test across held-out units.
"""
import argparse
import collections
import json
import math
import random
import re
import zipfile
import xml.etree.ElementTree as ET

import torch
from torch import nn

SPECIES = ('trigonelline', 'caffeine', '5-CQA', 'TDS')
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def xlsx_rows(path):
    """First worksheet as {column letter: value} dicts (stdlib only)."""
    z = zipfile.ZipFile(path)
    shared = [''.join(t.text or '' for t in si.iter(NS+'t'))
              for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall(NS+'si')]
    for r in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).iter(NS+'row'):
        row = {}
        for c in r.findall(NS+'c'):
            v = c.find(NS+'v')
            if v is not None:
                row[re.match(r'[A-Z]+', c.get('r')).group(0)] = shared[int(v.text)] if c.get('t') == 's' else v.text
        yield row


def load(path):
    shots, settings, rep = collections.OrderedDict(), None, None
    for row in xlsx_rows(path):
        if not row.get('G', '').isdigit():
            continue
        if row.get('B'):
            settings = {'condition': row['B'], 'flow': float(row['C']), 'grind': row['D'], 'temp': float(row['E'])}
        rep = row.get('F', rep)
        shot = shots.setdefault((settings['condition'], rep), {**settings, 'rep': rep, 'fractions': []})
        conc = [float(row[c]) if c in row else math.nan for c in 'JKLM']  # missing stays missing (masked)
        conc[3] *= 1000  # TDS g/g -> mg/g like the others
        shot['fractions'].append({'number': int(row['G']), 'mass': float(row['H']) if 'H' in row else None,
                                  'mid': float(row['I']) if 'I' in row else None, 'conc': conc})
    for shot in shots.values():
        fill_mass(shot['fractions'])
        shot['blocks'] = blocks(shot['fractions'])
    return list(shots.values())


def fill_mass(fractions):
    """A missing fraction mass between contiguous neighbours is fixed by their boundaries (no other imputation)."""
    for i, f in enumerate(fractions):
        if f['mass'] is None:
            prev, nxt = fractions[i-1], fractions[i+1]
            assert nxt['number']-prev['number'] == 2, 'missing mass is only determined between contiguous fractions'
            end, start = prev['mid']+prev['mass']/2, nxt['mid']-nxt['mass']/2
            f['mass'], f['mid'] = start-end, (start+end)/2


def blocks(fractions):
    """Percolation blocks (mass g, observed fraction index or None); gaps are unanalysed fractions."""
    out, end, number = [], 0., 0
    for i, f in enumerate(fractions):
        start = f['mid']-f['mass']/2
        if f['number'] > number+1:
            out.append((max(start-end, .01), None))
        out.append((f['mass'], i))
        end, number = start+f['mass'], f['number']
    return out


def features(shots):
    return torch.tensor([[s['flow']-2, (float(s['grind'])-1.7)/.3, (s['temp']-89)/9] for s in shots], dtype=torch.float64)


def design(x):
    return torch.cat([torch.ones(len(x), 1, dtype=torch.float64), x], 1)


def observed(shots):
    logc = torch.tensor([[[math.log(c) for c in f['conc']] for f in s['fractions']] for s in shots], dtype=torch.float64)
    mid = torch.tensor([[f['mid'] for f in s['fractions']] for s in shots], dtype=torch.float64)
    return logc, mid


def masked_mse(pred, y):
    ok = ~torch.isnan(y)
    return ((pred-y.nan_to_num())**2*ok).sum()/ok.sum()


def lbfgs(loss_fn, size, l2=1e-3):
    w = torch.zeros(size, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=500, tolerance_grad=1e-10, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = loss_fn(w)+l2*(w[1:]**2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach()


def curve_fit(train, test):
    xt, (y, mid), k = features(train), observed(train), len(SPECIES)

    def predict(w, x, mid):
        w = w.view(2, k, 4)
        tau = torch.exp(design(x) @ w[1].T+math.log(15))
        return (design(x) @ w[0].T)[:, None, :]-mid[..., None]/tau[:, None, :]
    w = lbfgs(lambda w: masked_mse(predict(w, xt, mid), y), 2*k*4)
    return predict(w, features(test), observed(test)[1])


def curve2_fit(train, test):
    xt, (y, mid), k = features(train), observed(train), len(SPECIES)

    def predict(w, x, mid):
        w, d, m = w.view(4, k, 4), design(x), mid[..., None]
        logc0, share = d @ w[0].T, torch.sigmoid(d @ w[1].T)
        fast, slow = torch.exp(d @ w[2].T+math.log(8)), torch.exp(d @ w[3].T+math.log(30))
        return logc0[:, None]+torch.log(share[:, None]*torch.exp(-m/fast[:, None])+(1-share[:, None])*torch.exp(-m/slow[:, None]))
    w = lbfgs(lambda w: masked_mse(predict(w, xt, mid), y), 4*k*4)
    return predict(w, features(test), observed(test)[1])


def block_inputs(shots):
    """(shots, blocks, 5): settings + block mass/5 + cumulative mass at block start/20."""
    x, out = features(shots), []
    for i, s in enumerate(shots):
        start, row = 0., []
        for mass, _ in s['blocks']:
            row.append(torch.cat([x[i], torch.tensor([mass/5, start/20], dtype=torch.float64)]))
            start += mass
        out.append(torch.stack(row))
    return torch.stack(out).float()


class Neural(nn.Module):
    def __init__(self, kind, hidden, mean_logc):
        super().__init__()
        self.kind, k = kind, len(SPECIES)
        if kind == 'mlp':
            self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, k))
            out = self.net[-1]
        else:
            self.start, self.gru = nn.Linear(3, hidden), nn.GRU(5, hidden, batch_first=True)
            out = self.head = nn.Linear(hidden, k)
        with torch.no_grad():
            out.bias.copy_(mean_logc.float())

    def forward(self, shots):
        if self.kind == 'mlp':
            _, mid = observed(shots)
            x = torch.cat([features(shots)[:, None].expand(-1, mid.shape[1], -1), (mid/20)[..., None]], 2).float()
            return self.net(x).double()
        hs, _ = self.gru(block_inputs(shots), torch.tanh(self.start(features(shots).float()))[None])
        keep = [b for b, (_, f) in enumerate(shots[0]['blocks']) if f is not None]
        return self.head(hs[:, keep]).double()


def neural_fit(train, test, kind, seed, epochs=300, hidden=16, lr=.01):
    y, _ = observed(train)
    ok = ~torch.isnan(y)
    torch.manual_seed(seed)
    model = Neural(kind, hidden, (y.nan_to_num()*ok).sum((0, 1))/ok.sum((0, 1)))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        masked_mse(model(train), y).backward()
        opt.step()
    with torch.no_grad():
        return model(test)


def folds(shots, split):
    if split == 'condition':
        for c in sorted({s['condition'] for s in shots}, key=int):
            yield c, [s for s in shots if s['condition'] != c], [s for s in shots if s['condition'] == c]
        return
    for key, levels in (('flow', (1., 3.)), ('grind', ('1.4', '2')), ('temp', (80., 98.))):
        for level in levels:
            yield f'{key}={level}', [s for s in shots if s[key] != level], [s for s in shots if s[key] == level]


def paired(errors, a, b, units, flips=20000):
    """Mean error difference b - a, wins for b, Monte Carlo sign-flip p."""
    d = [errors[b][u]-errors[a][u] for u in units]
    rng, observed_sum = random.Random(0), abs(sum(d))
    hits = sum(abs(sum(x*rng.choice((1, -1)) for x in d)) >= observed_sum-1e-12 for _ in range(flips))
    return {'mean_diff': sum(d)/len(d), 'wins': sum(x < 0 for x in d), 'of': len(d), 'p': hits/flips}


def run(shots, split, seeds, epochs):
    err, units = collections.defaultdict(dict), []
    for fold, train, test in folds(shots, split):
        y, _ = observed(test)
        ok = ~torch.isnan(y)
        preds = {'curve': [curve_fit(train, test)], 'curve2': [curve2_fit(train, test)]}
        for kind in ('mlp', 'gru'):
            preds[kind] = [neural_fit(train, test, kind, s, epochs) for s in seeds]
        for c in sorted({s['condition'] for s in test}, key=int):
            rows = torch.tensor([s['condition'] == c for s in test])
            unit = c if split == 'condition' else f'{fold}|{c}'
            units.append(unit)
            for arm, ps in preds.items():
                e = torch.stack([((p[rows]-y[rows].nan_to_num()).abs()*ok[rows]).sum((0, 1))/ok[rows].sum((0, 1)) for p in ps])
                err[arm][unit] = e.mean().item()
    return units, err


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='path to TableS1_ExperimentRawData.xlsx')
    p.add_argument('--split', choices=['condition', 'extrapolate'], default='condition')
    p.add_argument('--seeds', type=int, default=6)
    p.add_argument('--epochs', type=int, default=300)
    args = p.parse_args()
    units, err = run(load(args.data), args.split, tuple(range(args.seeds)), args.epochs)
    print(json.dumps({'split': args.split, 'held_out_units': len(units),
                      'log_mae': {a: round(sum(v.values())/len(v), 4) for a, v in err.items()},
                      'curve2_vs_curve': paired(err, 'curve', 'curve2', units),
                      'mlp_vs_curve2': paired(err, 'curve2', 'mlp', units)}, indent=1))


if __name__ == '__main__':
    main()
