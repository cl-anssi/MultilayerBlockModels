import argparse
import json
import os
import time

from collections import defaultdict

import numpy as np
import pandas as pd

from model import fit_mlplbm




parser = argparse.ArgumentParser()
parser.add_argument(
	'--input_file',
	required=True,
	help='Path to the input CSV file.' \
		+ 'Each line of the file represents one edge with the three' \
		+ 'following fields: top_node, bottom_node, edge_type.'
)
parser.add_argument(
	'--output_dir',
	default=None,
	help='Path to the directory where the results should be written.' \
		+ 'If None, then the results are written in the current' \
		+ 'working directory.'
)
parser.add_argument(
	'--H',
	type=int,
	nargs='+',
	default=list(range(2, 17)),
	help='Possible values for the number of top clusters.'
)
parser.add_argument(
	'--K',
	type=int,
	nargs='+',
	default=list(range(2, 17)),
	help='Possible values for the number of bottom clusters.'
)
parser.add_argument(
	'--runs',
	type=int,
	default=50,
	help='Number of runs of the inference procedure to perform for' \
		+ 'each possible model.' \
		+ 'Different initial parameters are used for each run, and' \
		+ 'the best-performing model is returned.'
)
parser.add_argument(
	'--epsilon',
	type=float,
	default=1e-7,
	help='Stopping criterion for the inference procedure.'
)
parser.add_argument(
	'--max_iter',
	type=int,
	default=2000,
	help='Maximum number of iterations for the inference procedure.'
)
parser.add_argument(
	'--jobs',
	type=int,
	default=1,
	help='Number of parallel workers for the inference procedure.'
)
parser.add_argument(
	'--backend',
	default='numpy',
	help='Backend to use for the computations.' \
		+ 'Possible values: numpy, torch.'
)
parser.add_argument(
	'--device',
	nargs='+',
	default=['cuda'],
	help='Identifiers of the devices used by PyTorch.' \
		+ 'If the number of devices is greater than the number of' \
		+ 'jobs, than only the first n_jobs devices are used.'
)
parser.add_argument(
	'--verbose',
	type=int,
	default=1,
	help='Level of verbosity (0, 1 or >1).'
)
parser.add_argument(
	'--seed',
	type=int,
	default=None,
	help='Random seed for the RNG.' \
		+ 'If None, the seed is not set.'
)
args = parser.parse_args()


df = pd.read_csv(args.input_file)
X = df.to_numpy()
est = fit_mlplbm(
	X,
	H=args.H,
	K=args.K,
	runs=args.runs,
	epsilon=args.epsilon,
	max_iter=args.max_iter,
	verbose=args.verbose,
	backend=args.backend,
	device=args.device,
	n_jobs=args.jobs,
	random_state=args.seed
)

top, bottom, thetas = est.get_results()

top_clust = dict(
	zip(
		[u for x in top for u in x],
		[h for h, x in enumerate(top) for i in x]
	)
)
bottom_clust = dict(
	zip(
		[v for x in bottom for v in x],
		[k for k, x in enumerate(bottom) for j in x]
	)
)

df['top_clust'] = df['top'].apply(lambda x: top_clust[x])
df['bottom_clust'] = df['bottom'].apply(lambda x: bottom_clust[x])
cnt = df.groupby(['top_clust', 'bottom_clust', 'type']).sum()
tmp = cnt.to_dict()['count']
counts = {}
for t in thetas:
	counts[t] = [
		[
			tmp[(h, k, t)] if (h, k, t) in tmp else 0
			for k in range(est.K)
		]
		for h in range(est.H)
	]

res = {
	'top': top,
	'bottom': bottom,
	'thetas': thetas,
	'counts': counts
}

fname = 'results_%d.json' % int(time.time())
if args.output_dir is not None:
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	fp = os.path.join(args.output_dir, fname)
else:
	fp = fname
with open(fp, 'w') as out:
	out.write(json.dumps(res))
