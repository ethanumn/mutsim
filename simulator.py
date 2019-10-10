import numpy as np
from collections import defaultdict, OrderedDict
import common

def _make_parents(K):
  # Determine parents of nodes [1, 2, ..., K].
  parents = []
  # mu is the probability of extending the current branch.
  mu = 0.75
  for idx in range(K - 1):
    U = np.random.uniform()
    if U < mu:
      parents.append(len(parents))
    else:
      parents.append(np.random.randint(0, idx + 1))
  return np.array(parents)

def make_parents(K, tree_type):
  assert tree_type in (None, 'monoprimary', 'polyprimary')
  while True:
    parents = _make_parents(K)
    root_children = np.sum(parents == 0)
    if tree_type is None:
      break
    elif tree_type == 'monoprimary' and root_children == 1:
      assert parents[0] == 0 and not np.any(parents[1:] == 0)
      break
    elif tree_type == 'polyprimary' and root_children > 1:
      break

  return parents

def _make_adjm(parents):
  K = len(parents) + 1
  adjm = np.eye(K)
  adjm[parents, range(1, K)] = 1
  return adjm

def generate_tree(K, S, tree_type):
  parents = make_parents(K, tree_type)
  #leaves = np.flatnonzero(np.sum(adjm, axis=1) == 0)
  adjm = _make_adjm(parents)
  Z = common.make_ancestral_from_adj(adjm) # KXK

  eta = np.random.dirichlet(alpha = K*[1e0], size = S).T # KxS
  # In general, we want etas on leaves to be more "peaked" -- that is, only a
  # few subclones come to dominate, so they should have large etas relative to
  # internal nodes. We accomplish this by using a smaller alpha for these.
  #eta[leaves] += np.random.dirichlet(alpha = len(leaves)*[1e0], size = S).T
  eta /= np.sum(eta, axis=0)

  phi = np.dot(Z, eta) # KxS
  assert np.allclose(1, phi[0])
  return (parents, phi)

def generate_read_counts(phi, omega_v, T):
  M, S = phi.shape
  # T: total reads. Broadcast operation ensures V and T are same shape.
  T = np.broadcast_to(T, (M,S))
  # V: variant reads
  V = np.random.binomial(n=T, p=omega_v*phi)
  return (V, T)

def add_noise(mat, sigma=0.09):
  noisy = np.random.normal(loc=mat, scale=sigma)
  capped = np.maximum(0, np.minimum(1, noisy))
  return capped

def choose_categories(C, alpha, size=None):
  P = np.random.dirichlet(alpha = C*[alpha])
  return np.random.choice(C, p=P, size=size)

def assign_ssms(K, M):
  # Ensure every cluster has at least one mutation.
  assert M >= K
  first_ssmass = np.arange(K)
  probs = np.random.dirichlet(alpha=K*[1])
  remaining_ssmass = np.random.choice(K, p=probs, size=(M - K))
  ssmass = np.concatenate((first_ssmass, remaining_ssmass))
  np.random.shuffle(ssmass)
  # Add one so that no SSMs are assigned to the root.
  return ssmass + 1

def make_clusters(ssmass):
  clusters = defaultdict(list)
  for midx, cidx in enumerate(ssmass):
    clusters[cidx].append('s%s' % midx)
  assert set(clusters.keys()) == set(range(1, len(clusters) + 1))

  clusters = [clusters[cidx] for cidx in sorted(clusters.keys())]
  return clusters

def make_variants(phi_mutations, T, omega_v):
  V, T = generate_read_counts(phi_mutations, omega_v, T)

  variants = OrderedDict()
  for midx in range(len(omega_v)):
    variant = {
      'id': 's%s' % midx,
      'name': 'S_%s' % midx,
      'var_reads': V[midx],
      'total_reads': T[midx],
      'omega_v': omega_v[midx],
      'phi': phi_mutations[midx],
    }
    variant['ref_reads'] = variant['total_reads'] - variant['var_reads']
    variant['vaf'] = variant['var_reads'] / variant['total_reads']
    variants[variant['id']] = variant
  return variants

def segment_genome(H, alpha=5):
  masses = np.random.dirichlet(alpha = H*[alpha])
  return masses

def _create_cna_config(K, H, C, ploidy):
  cn_pop_probs = np.random.dirichlet(alpha = K*[5])
  cn_seg_probs = np.random.dirichlet(alpha = H*[5])
  cn_phase_probs = np.random.dirichlet(alpha = ploidy*[5])

  attempts = 0
  max_attempts = 5000*C
  events = set()

  while len(events) < C:
    attempts += 1
    if attempts > max_attempts:
      raise Exception('Could not generate configuration without duplicates in %s attempts' % max_attempts)
    # Add one so that no CNAs are assigned to the root.
    cn_pop = np.random.choice(K, p=cn_pop_probs) + 1
    cn_seg = np.random.choice(H, p=cn_seg_probs)
    cn_phase = np.random.choice(ploidy, p=cn_phase_probs)
    triplet = (cn_pop, cn_seg, cn_phase)
    if triplet not in events:
      events.add(triplet)

  combined = np.array(list(events)).T
  cn_pops, cn_segs, cn_phases = combined
  return (cn_pops, cn_segs, cn_phases)

def _find_children(adjm):
  adjm = np.copy(adjm)
  np.fill_diagonal(adjm, 0)
  adjl = {}
  for pidx in range(len(adjm)):
    # Convert to Python list so we don't get weird NumPy broadcasting
    # behaviour.
    adjl[pidx] = np.flatnonzero(adjm[pidx]).tolist()
  return adjl

def generate_cnas(K, C, segs, parents, prop_gains=0.8):
  ploidy = 2
  root = 0
  H = len(segs)
  children = _find_children(adjm)

  cn_pops, cn_segs, cn_phases = _create_cna_config(K, H, C, ploidy)
  alleles = np.nan * np.ones((K+1, H, ploidy))
  alleles[root,:,:] = 1

  del_idxs = np.random.uniform(size=C) >= prop_gains
  # Take deltas in integer range [1, 2, ...].
  lam = 1.5
  cn_deltas = np.ceil(np.random.exponential(scale=1/lam, size=C))
  assert np.all(cn_deltas >= 1)
  cn_deltas[del_idxs] *= -1

  stack = list(children[root])
  while len(stack) > 0:
    pop = stack.pop()
    pop_cna = np.flatnonzero(cn_pops == pop)
    parent = parents[pop]
    alleles[pop] = alleles[parent]

    for cna in pop_cna:
      parent_cn = alleles[parent, cn_segs[cna], cn_phases[cna]]
      assert parent_cn >= 0

      # Once an allele disappears, don't permit it to return.
      if parent_cn == 0:
        # Note that the only instance in which cn_deltas can be zero is when a
        # parent in the tree has already dropped this segment's CN to zero.
        cn_deltas[cna] = 0
      # Never let CN drop below zero.
      if cn_deltas[cna] < 0:
        cn_deltas[cna] = np.maximum(cn_deltas[cna], -parent_cn)
      alleles[pop, cn_segs[cna], cn_phases[cna]] = parent_cn + cn_deltas[cna]

    stack += children[pop]

  assert not np.any(np.isnan(alleles))
  assert np.all(alleles >= 0)
  return (cn_pops, cn_segs, cn_phases, cn_deltas, alleles)

def generate_data(K, S, T, M, C, H, G, tree_type):
  # K: number of clusters (excluding normal root)
  # S: number of samples
  # T: reads per mutation
  # M: total number of SSMs
  # C: total number of CNAs
  # H: number of genomic segments
  # G: number of (additional) garbage mutations
  parents, phi = generate_tree(K + 1, S, tree_type)
  # Add 1 to each mutation's assignment to account for normal root.
  ssmass = assign_ssms(K, M) # Mx1
  clusters = make_clusters(ssmass)

  phi_good_mutations = np.array([phi[cidx] for cidx in ssmass]) # MxS
  phi_garbage = np.random.uniform(size=(G,S))
  phi_mutations = np.vstack((phi_good_mutations, phi_garbage))

  segs = segment_genome(H)
  #cn_pops, cn_segs, cn_phases, cn_deltas, alleles = generate_cnas(K, C, segs, adjm)

  omega_v = np.broadcast_to(0.5, (M + G, S))
  variants = make_variants(phi_mutations, T, omega_v)
  vids_good = ['s%s' % vidx for vidx in range(M)]
  vids_garbage = ['s%s' % vidx for vidx in range(M, M + G)]
  assert set(vids_good) == set([V for C in clusters for V in C])

  return {
    'sampnames': ['Sample %s' % (sidx + 1) for sidx in range(S)],
    'structure': parents,
    'phi': phi,
    'clusters': clusters,
    'variants': variants,
    'vids_good': vids_good,
    'vids_garbage': vids_garbage,
    'segments': segs,
    #'cn_pops': cn_pops,
    #'cn_segs': cn_segs,
    #'cn_phases': cn_phases,
    #'cn_deltas': cn_deltas,
    #'alleles': alleles,
  }
