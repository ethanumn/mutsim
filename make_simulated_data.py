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

def write_full_data(data, datafn):
  with open(datafn, 'wb') as outf:
    pickle.dump(data, outf)

def write_params(data, paramsfn, should_write_clusters):
  with open(paramsfn, 'w') as outf:
    params = {
      'samples': data['sampnames'],
    }
    if should_write_clusters:
      params['clusters'] = data['clusters']
      params['garbage'] = data['vids_garbage']
    json.dump(params, outf)

def main():
  np.set_printoptions(linewidth=400, precision=3, threshold=sys.maxsize, suppress=True)
  parser = argparse.ArgumentParser(
    description='LOL HI THERE',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument('--seed', dest='seed', type=int)
  parser.add_argument('--write-clusters', action='store_true')
  parser.add_argument('--tree-type', choices=('monoprimary', 'polyprimary'))
  parser.add_argument('--write-ssm-phi', action='store_true')
  parser.add_argument('-K', dest='K', type=int, default=4, help='Number of clusters')
  parser.add_argument('-S', dest='S', type=int, default=3, help='Number of samples')
  parser.add_argument('-T', dest='T', type=int, default=4000, help='Total reads per mutation')
  parser.add_argument('-M', dest='M', type=int, default=10, help='Number of non-garbage mutations')
  parser.add_argument('-C', dest='C', type=int, default=50, help='Number of CN events')
  parser.add_argument('-H', dest='H', type=int, default=30, help='Number of genomic segments')
  parser.add_argument('-G', dest='G', type=int, default=4, help='Number of garbage mutations')
  parser.add_argument('datafn')
  parser.add_argument('paramsfn')
  parser.add_argument('ssmfn')
  args = parser.parse_args()

  if args.seed is None:
    seed = np.random.randint(2**31)
  else:
    seed = args.seed
  np.random.seed(args.seed)

  data = simulator.generate_data(
    args.K,
    args.S,
    args.T,
    args.M,
    args.C,
    args.H,
    args.G,
    args.tree_type
  )
  data['seed'] = seed

  write_full_data(data, args.datafn)
  write_params(data, args.paramsfn, args.write_clusters)
  write_ssms(data, args.ssmfn, args.write_ssm_phi)

if __name__ == '__main__':
  main()
