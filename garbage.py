import argparse
import numpy as np
import json
import sys
import os

import simulator
sys.path.append(os.path.expanduser('~/work/pairtree/lib'))
import util
from common import Models

def _add_uniform(G, S):
  phi_garb = np.random.uniform(size=(G,S))
  return phi_garb

def _add_missed_cna(G, phi_good, struct, omega=1.):
  K, S = phi_good.shape
  K -= 1 # Don't count root node.
  indices = np.random.choice(np.arange(1, K+1), size=G, replace=True)
  phi_garb = phi_good[indices]
  omega_true = np.broadcast_to(1.0, (G,S))
  omega_observed = np.broadcast_to(0.5, (G,S))
  return (phi_garb, omega_true, omega_observed)

def _make_noderels(struct):
  adjm = util.convert_parents_to_adjmatrix(struct)
  rels = util.compute_node_relations(adjm)
  return rels

def _choose_subset(P, Q):
  '''
  Sample `Q` indices from {0, 1, ..., P-1} without replacement.
  '''
  return np.random.choice(np.arange(P), size=Q, replace=False)

def _sample_node_pair_in_relation(K, rel, noderels):
  assert K >= 2
  while True:
    # Sample indices from [1, 2, ..., K].
    A, B = np.random.choice(np.arange(1, K+1), size=2, replace=False)
    if noderels[A,B] != rel:
      continue
    return (A, B)

def _add_acquired_twice(G, phi_good, struct):
  noderels = _make_noderels(struct)
  phi_garb = []
  K = len(phi_good) - 1

  while len(phi_garb) < G:
    A, B = _sample_node_pair_in_relation(K, Models.diff_branches, noderels)
    if np.all(phi_good[A] < 0.5) or np.all(phi_good[B] < 0.5):
      continue
    phi = phi_good[A] + phi_good[B]
    if np.any(phi >= 1.):
      continue
    print('merging', A, B, len(phi_garb) + len(phi))
    print(np.array([phi_good[A], phi_good[B], phi]))
    phi_garb.append(phi)

  return phi_garb

def _add_wildtype_backmut(G, phi_good, struct):
  noderels = _make_noderels(struct)
  phi_garb = []
  K = len(phi_good) - 1

  while len(phi_garb) < G:
    A, B = _sample_node_pair_in_relation(K, Models.A_B, noderels)
    assert np.all(phi_good[A] >= phi_good[B])
    phi = phi_good[A] - phi_good[B]
    phi_garb.append(phi)

  return phi_garb

def generate(G, garbage_type, struct, phi_good):
  _, S = phi_good.shape
  omega_diploid = np.broadcast_to(0.5, (G,S))
  omega_true = omega_observed = omega_diploid

  if garbage_type == 'uniform':
    phi_garb = _add_uniform(G, S)
  elif garbage_type == 'acquired_twice':
    phi_garb = _add_acquired_twice(G, phi_good, struct)
  elif garbage_type == 'wildtype_backmut':
    phi_garb = _add_wildtype_backmut(G, phi_good, struct)
  elif garbage_type == 'missed_cna':
    # Overwrite current `omega_true` and `omega_observed`.
    phi_garb, omega_true, omega_observed = _add_missed_cna(G, phi_good, struct)
  else:
    raise Exception('Unknown garbage type')

  return (phi_garb, omega_true, omega_observed)
