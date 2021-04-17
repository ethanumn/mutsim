import argparse
import numpy as np
import json
import sys
import os

import simulator
sys.path.append(os.path.expanduser('~/work/pairtree/lib'))
import util
from common import Models

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

def _is_garbage(phi_garb, phi_good):
  K, S = phi_good.shape
  assert phi_garb.shape == (S,)
  for other in phi_good:
    has_AB = np.any(other > phi_garb)
    has_BA = np.any(other < phi_garb)
    has_branched = np.any(other + phi_garb > 1)
    if has_AB and has_BA and has_branched:
      return True
  return False

def _add_uniform(phi_good, struct):
  K, S = phi_good.shape
  phi = np.random.uniform(size=S)
  omega_diploid = np.broadcast_to(0.5, S)
  return (phi, omega_diploid, omega_diploid)

def _add_missed_cna(phi_good, struct, omega=1.):
  K, S = phi_good.shape
  K -= 1 # Correct for root node.

  idx = np.random.choice(np.arange(1, K+1))
  phi = phi_good[idx]
  omega_true = np.broadcast_to(omega, S)
  omega_observed = np.broadcast_to(0.5, S)
  return (phi, omega_true, omega_observed)

def _add_acquired_twice(phi_good, struct):
  K, S = phi_good.shape
  K -= 1 # Correct for root node.
  noderels = _make_noderels(struct)

  A, B = _sample_node_pair_in_relation(K, Models.diff_branches, noderels)
  phi = phi_good[A] + phi_good[B]
  assert np.all(phi <= 1.)
  omega_diploid = np.broadcast_to(0.5, S)
  return (phi, omega_diploid, omega_diploid)

def _add_wildtype_backmut(phi_good, struct):
  K, S = phi_good.shape
  K -= 1 # Correct for root node.

  noderels = _make_noderels(struct)
  A, B = _sample_node_pair_in_relation(K, Models.A_B, noderels)
  phi = phi_good[A] - phi_good[B]
  assert np.all(phi >= 0)
  omega_diploid = np.broadcast_to(0.5, S)
  return (phi, omega_diploid, omega_diploid)

class TooManyAttemptsError(Exception):
  pass

def generate(G, garbage_type, struct, phi_good, max_attempts=1000):
  _, S = phi_good.shape
  omega_diploid = np.broadcast_to(0.5, (G,S))
  omega_true = omega_observed = omega_diploid

  if garbage_type == 'uniform':
    gen_garb = _add_uniform
  elif garbage_type == 'acquired_twice':
    gen_garb = _add_acquired_twice
  elif garbage_type == 'wildtype_backmut':
    gen_garb = _add_wildtype_backmut
  elif garbage_type == 'missed_cna':
    # Overwrite current `omega_true` and `omega_observed`.
    gen_garb = _add_missed_cna
  else:
    raise Exception('Unknown garbage type')

  attempts = 0
  phi_garb = []
  omega_true = []
  omega_observed = []

  while len(phi_garb) < G:
    attempts += 1
    if attempts > max_attempts:
      raise TooManyAttemptsError()
    phi, o_true, o_obs = gen_garb(phi_good, struct)
    if not _is_garbage(phi, phi_good):
      continue
    phi_garb.append(phi)
    omega_true.append(o_true)
    omega_observed.append(o_obs)

  return [np.array(A) for A in (phi_garb, omega_true, omega_observed)]
