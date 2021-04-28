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
    # Sometimes the tree will not permit this. Raise an exception in this case.
    if not np.any(noderels == rel):
      raise simulator.TreeDoesNotSatisfyRelationsError()
    A, B = np.random.choice(np.arange(1, K+1), size=2, replace=False)
    if noderels[A,B] != rel:
      continue
    A_muts, B_muts = _find_mut_indices(A, ssm_ass), _find_mut_indices(B, ssm_ass)
    assert len(A_muts) > 0 and len(B_muts) > 0

    A_mut, B_mut = np.random.choice(A_muts), np.random.choice(B_muts)
    assert A_mut != B_mut
    return (A_mut, B_mut)

def _is_garbage(phi_garb, omega_garb_true, omega_garb_obs, phi_good, omega_good, min_garb_phi_delta=0.1, min_garb_samps=3, min_garb_pairs=3):
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
  garb_pairs = 0
  for other in phi_good:
    cannot_be = {
      # phi_garb doesn't allow it to be on different branch relative to other
      'branched': other + phi_hat_garb >= 1 + min_garb_phi_delta,
      # phi_garb doesn't allow it to be ancestor of other
      'AB':       other - phi_hat_garb >= min_garb_phi_delta,
      # phi_garb doesn't allow it to be descendant of other
      'BA':       phi_hat_garb - other >= min_garb_phi_delta,
    }
    num_cannot = {K: np.sum(V) for K,V in cannot_be.items()}

    if np.all(np.array(list(num_cannot.values())) >= min_garb_samps):
      #print(num_cannot, phi_hat_garb, phi_garb, min_garb_phi_delta, min_garb_samps)
      garb_pairs += 1
      if garb_pairs >= min_garb_pairs:
        return True
  return False

def _add_uniform(phi_good, omega_good, struct, ssm_ass):
  _, S = phi_good.shape
  phi = np.random.uniform(size=S)
  omega_diploid = np.broadcast_to(0.5, S)
  return (phi, omega_diploid, omega_diploid)

def _add_missed_cna(phi_good, omega_good, struct, ssm_ass, omega_true=1., omega_obs=0.5, make_obvious=False, min_delta=0.01):
  M, S = phi_good.shape

  if make_obvious:
    phi_good_threshold = 0.5
    P = np.zeros(M)
    P[np.any(phi_good > phi_good_threshold + min_delta, axis=1)] = 1.
    if np.all(P == 0):
      raise simulator.NoBigEnoughPhiError()
    else:
      P /= np.sum(P)
  else:
    P = np.ones(M) / M
  idx = np.random.choice(M, p=P)

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

def _ensure_between(A, lower, upper):
  under = A < lower
  over = A > upper

  if np.any(under):
    assert np.allclose(lower, A[under])
    A[under] = lower
  if np.any(over):
    assert np.allclose(upper, A[over])
    A[over] = upper
  assert np.all(A >= lower) and np.all(A <= upper)

def generate(G, garbage_type, min_garb_pairs, min_garb_phi_delta, min_garb_samps, make_missed_cna_obvious, struct, phi_good_muts, omega_good, ssm_ass, max_attempts=100):
  def __add_missed_cna(*args):
    return _add_missed_cna(*args, make_obvious=make_missed_cna_obvious, min_delta=min_garb_phi_delta)

  if garbage_type == 'uniform':
    gen_garb = _add_uniform
  elif garbage_type == 'acquired_twice':
    gen_garb = _add_acquired_twice
  elif garbage_type == 'wildtype_backmut':
    gen_garb = _add_wildtype_backmut
  elif garbage_type == 'missed_cna':
    gen_garb = __add_missed_cna
  else:
    raise Exception('Unknown garbage type')

  attempts = 0
  phi_garb = []
  omega_true = []
  omega_observed = []

  _, S = phi_good_muts.shape

  while len(phi_garb) < G:
    attempts += 1
    if attempts > max_attempts:
      raise simulator.TooManyAttemptsError()
    phi, o_true, o_obs = gen_garb(phi_good_muts, omega_good, struct, ssm_ass)
    for A in (phi, o_true, o_obs):
      _ensure_between(A, 0., 1.)
    if not _is_garbage(phi, o_true, o_obs, phi_good_muts, omega_good, min_garb_phi_delta=min_garb_phi_delta, min_garb_samps=min_garb_samps, min_garb_pairs=min_garb_pairs):
      continue

    phi_garb.append(phi)
    omega_true.append(o_true)
    omega_observed.append(o_obs)

  return [np.array(A) for A in (phi_garb, omega_true, omega_observed)]
