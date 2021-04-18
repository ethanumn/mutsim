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

def _find_mut_indices(kidx, ssm_ass):
  return [idx for idx, ass in enumerate(ssm_ass) if ass == kidx]

def _sample_mut_pair_in_relation(rel, struct, ssm_ass):
  K = len(struct)
  noderels = _make_noderels(struct)

  while True:
    # Sample indices from [1, 2, ..., K].
    A, B = np.random.choice(np.arange(1, K+1), size=2, replace=False)
    if noderels[A,B] != rel:
      continue
    A_muts, B_muts = _find_mut_indices(A, ssm_ass), _find_mut_indices(B, ssm_ass)
    assert len(A_muts) > 0 and len(B_muts) > 0

    A_mut, B_mut = np.random.choice(A_muts), np.random.choice(B_muts)
    assert A_mut != B_mut
    return (A_mut, B_mut)

def _is_garbage(phi_garb, omega_garb_true, omega_garb_obs, phi_good, omega_good):
  _, S = phi_good.shape
  assert phi_garb.shape == (S,)

  # We must test with phi_hat, not with phi, since these are the values
  # Pairtree will see. A legitimate garbage mutation may be rendered garbage
  # only by its implied phi_hat, not its true phi. For every case except
  # "missed CNA", we will have phi_hat = phi, since omega_true = omega_obs =
  # 0.5. But for "missed CNA", this is important.
  allelefrac_garb = phi_garb * omega_garb_true
  phi_hat_garb = allelefrac_garb / omega_garb_obs
  phi_hat_garb = np.minimum(1., phi_hat_garb)

  # Ensure that a putative garbage mutation is rendered garbage with respect to
  # at least one legitimate mutation.
  for other in phi_good:
    has_AB = np.any(other > phi_hat_garb)
    has_BA = np.any(other < phi_hat_garb)
    has_branched = np.any(other + phi_hat_garb > 1)
    if has_AB and has_BA and has_branched:
      print(phi_hat_garb, phi_garb)
      return True
  return False

def _add_uniform(phi_good, omega_good, struct, ssm_ass):
  _, S = phi_good.shape
  phi = np.random.uniform(size=S)
  omega_diploid = np.broadcast_to(0.5, S)
  return (phi, omega_diploid, omega_diploid)

def _add_missed_cna(phi_good, omega_good, struct, ssm_ass, omega_true=1., omega_obs=0.5):
  M, S = phi_good.shape

  idx = np.random.choice(np.arange(1, M))
  assert np.allclose(0.5, omega_good[idx])
  phi = phi_good[idx]
  omega_true = np.broadcast_to(omega_true, S)
  omega_obs = np.broadcast_to(omega_obs, S)
  return (phi, omega_true, omega_obs)

def _add_acquired_twice(phi_good, omega_good, struct, ssm_ass):
  K = len(struct)
  _, S = phi_good.shape

  A, B = _sample_mut_pair_in_relation(Models.diff_branches, struct, ssm_ass)
  phi = phi_good[A] + phi_good[B]
  assert np.all(phi <= 1.)
  omega_diploid = np.broadcast_to(0.5, S)
  return (phi, omega_diploid, omega_diploid)

def _add_wildtype_backmut(phi_good, omega_good, struct, ssm_ass):
  K = len(struct)
  _, S = phi_good.shape

  A, B = _sample_mut_pair_in_relation(Models.A_B, struct, ssm_ass)
  phi = phi_good[A] - phi_good[B]
  assert np.all(phi >= 0)
  omega_diploid = np.broadcast_to(0.5, S)
  return (phi, omega_diploid, omega_diploid)

class TooManyAttemptsError(Exception):
  pass

def generate(G, garbage_type, struct, phi_good_muts, omega_good, ssm_ass, max_attempts=10000):
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
    phi, o_true, o_obs = gen_garb(phi_good_muts, omega_good, struct, ssm_ass)
    if not _is_garbage(phi, o_true, o_obs, phi_good_muts, omega_good):
      continue
    phi_garb.append(phi)
    omega_true.append(o_true)
    omega_observed.append(o_obs)

  return [np.array(A) for A in (phi_garb, omega_true, omega_observed)]
