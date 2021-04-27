from __future__ import print_function
import argparse
import numpy as np
import pickle
import json
import sys

import simulator

def write_ssms(data, ssm_fn, write_ssm_phi):
  fields = ['id', 'name', 'var_reads', 'total_reads', 'var_read_prob']
  vector_fields = ['var_reads', 'total_reads', 'omega_v']
  if write_ssm_phi:
    fields.append('phi')
    vector_fields.append('phi')

  with open(ssm_fn, 'w') as outf:
    print(*fields, sep='\t', file=outf)
    for V in data['variants'].values():
      V = dict(V) # Don't modify original variant.
      for K in vector_fields:
        V[K] = ','.join([str(R) for R in V[K]])
      V['var_read_prob'] = V['omega_v']
      print(*[V[K] for K in fields], sep='\t', file=outf)

def write_full_data(simdata, simparams, truthfn):
  # Don't modify original dict.
  combined = dict(simdata)
  combined['simparams'] = simparams
  with open(truthfn, 'wb') as outf:
    pickle.dump(combined, outf)

def write_params(simdata, simparams, paramsfn, should_write_clusters, should_write_structures):
  with open(paramsfn, 'w') as outf:
    params = {
      'samples': simdata['sampnames'],
      'garbage': simdata['vids_garbage'],
      'simparams': simparams,
    }
    if should_write_clusters:
      params['clusters'] = simdata['clusters']
    if should_write_structures:
      params['structures'] = [simdata['structure'].tolist()]
    json.dump(params, outf)

def write_numpy(data, numpy_fn):
  arrs = simulator.convert_to_numpy_array(data)
  np.savez_compressed(numpy_fn, **arrs)

def main():
  np.set_printoptions(linewidth=400, precision=3, threshold=sys.maxsize, suppress=True)
  parser = argparse.ArgumentParser(
    description='LOL HI THERE',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument('--seed', dest='seed', type=int)
  parser.add_argument('--tree-type', choices=('monoprimary', 'polyprimary'))
  parser.add_argument('--write-clusters', action='store_true')
  parser.add_argument('--write-ssm-phi', action='store_true')
  parser.add_argument('--write-structures', action='store_true')
  parser.add_argument('--write-numpy', dest='numpy_fn')
  parser.add_argument('--alpha', type=float, default=0.1, help='Alpha parameter used for sampling eta from Dirichlet')
  parser.add_argument('-K', dest='K', type=int, default=4, help='Number of clusters')
  parser.add_argument('-S', dest='S', type=int, default=3, help='Number of samples')
  parser.add_argument('-T', dest='T', type=int, default=4000, help='Total reads per mutation')
  parser.add_argument('-M', dest='M', type=int, default=10, help='Number of non-garbage mutations')
  parser.add_argument('-C', dest='C', type=int, default=0, help='Number of CN events')
  parser.add_argument('-H', dest='H', type=int, default=1, help='Number of genomic segments')
  parser.add_argument('-G', dest='G', type=int, default=0, help='Number of garbage mutations')
  parser.add_argument('--garbage-type', choices=('acquired_twice', 'wildtype_backmut', 'uniform', 'missed_cna'), default='uniform')
  parser.add_argument('--make-missed-cna-obvious', action='store_true')
  parser.add_argument('--min-garb-pairs', type=int, default=3)
  parser.add_argument('--min-garb-phi-delta', type=float, default=0.1)
  parser.add_argument('--min-garb-samps', type=int, default=1)
  parser.add_argument('truthfn')
  parser.add_argument('paramsfn')
  parser.add_argument('ssmfn')
  args = parser.parse_args()

  if args.seed is None:
    seed = np.random.randint(2**31)
  else:
    seed = args.seed
  np.random.seed(args.seed)

  max_attempts = 10000
  attempts = 0
  while True:
    attempts += 1
    try:
      simdata, simparams = simulator.generate_data(
        args.K,
        args.S,
        args.T,
        args.M,
        args.C,
        args.H,
        args.G,
        args.garbage_type,
        args.min_garb_pairs,
        args.min_garb_phi_delta,
        args.min_garb_samps,
        args.make_missed_cna_obvious,
        args.alpha,
        args.tree_type
      )
      break
    except (simulator.TooManyAttemptsError, simulator.TreeDoesNotSatisfyRelationsError, simulator.NoBigEnoughPhiError):
      if attempts >= max_attempts:
        raise
      else:
        print('Failed to satisfy required conditions on attempt %s, retrying ...' % attempts)
        continue
  simparams['seed'] = seed
  simparams['args'] = dict(vars(args))

  write_full_data(simdata, simparams,args.truthfn)
  write_params(simdata, simparams, args.paramsfn, args.write_clusters, args.write_structures)
  write_ssms(simdata, args.ssmfn, args.write_ssm_phi)
  if args.numpy_fn is not None:
    write_numpy(simdata, args.numpy_fn)

if __name__ == '__main__':
  main()
