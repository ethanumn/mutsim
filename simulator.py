import numpy as np
from collections import defaultdict, OrderedDict, namedtuple
import common

Cna = namedtuple('Cna', ('pop', 'seg', 'phase', 'delta'))
TIMING_BEFORE = 0
TIMING_AFTER = 1
DIRECTION_GAIN = 1
DIRECTION_LOSS = 0

def _make_parents(K):
  # Determine parents of nodes [1, 2, ..., K].
  parents = []
  # mu is the probability of extending the current branch.
  mu = 0.75
  for idx in range(K):
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

def generate_tree(K, S, alpha, tree_type):
  parents = make_parents(K, tree_type)
  #leaves = np.flatnonzero(np.sum(adjm, axis=1) == 0)
  adjm = _make_adjm(parents)
  Z = common.make_ancestral_from_adj(adjm) # (K+1)x(K+1)

  eta = np.random.dirichlet(alpha = (K+1)*[alpha], size=S).T # (K+1)xS
  # In general, we want etas on leaves to be more "peaked" -- that is, only a
  # few subclones come to dominate, so they should have large etas relative to
  # internal nodes. We accomplish this by using a smaller alpha for these.
  #eta[leaves] += np.random.dirichlet(alpha = len(leaves)*[1e0], size = S).T
  eta /= np.sum(eta, axis=0)

  phi = np.dot(Z, eta) # (Kx1)xS
  assert np.allclose(1, phi[0])
  return (parents, phi, eta)

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

def assign_ssms_to_pops(M, K):
  # Ensure every cluster has at least one mutation.
  assert M >= K
  first_ssmass = np.arange(K)
  probs = np.random.dirichlet(alpha=K*[1])
  remaining_ssmass = np.random.choice(K, p=probs, size=(M - K))
  ssmass = np.concatenate((first_ssmass, remaining_ssmass))
  np.random.shuffle(ssmass)
  # Add one so that no SSMs are assigned to the root.
  return ssmass + 1

def assign_ssms_to_segs(M, segs):
  seg_ass = np.random.choice(len(segs), p=segs, size=M)
  return seg_ass

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
  segs = np.random.dirichlet(alpha = H*[5])
  return segs

def _generate_cna_events(K, H, C, ploidy, struct):
  assert len(struct) == K
  adjm = _make_adjm(struct)
  anc = common.make_ancestral_from_adj(adjm)

  cn_seg_probs = np.random.dirichlet(alpha = H*[5])
  cn_phase_probs = np.random.dirichlet(alpha = ploidy*[5])
  cn_pop_probs = np.random.dirichlet(alpha = K*[5])
  # Directions: 0=deletion, 1=gain
  direction_probs = np.random.dirichlet(alpha = 2*[5])
  lam = 1.5

  attempts = 0
  max_attempts = 5000*C

  events = []
  triplets = set()
  directions = {}
  deletions = {}

  while len(events) < C:
    attempts += 1
    if attempts > max_attempts:
      raise Exception('Could not generate configuration without duplicates in %s attempts' % max_attempts)

    cn_seg = np.random.choice(H, p=cn_seg_probs)
    cn_phase = np.random.choice(ploidy, p=cn_phase_probs)
    # Add one so that no CNAs are assigned to the root.
    cn_pop = np.random.choice(K, p=cn_pop_probs) + 1
    triplet = (cn_seg, cn_phase, cn_pop)
    doublet = (cn_seg, cn_phase)

    if triplet in triplets:
      continue

    if doublet in directions:
      direction = directions[doublet]
    else:
      direction = np.random.choice(2, p=direction_probs)

    if direction == DIRECTION_GAIN:
      delta = np.ceil(np.random.exponential(scale=1/lam)).astype(np.int)
      assert delta >= 1
    else:
      # We only ever have one allele to lose, so can never lose more than one.
      delta = -1
      if doublet in deletions:
        same_branch_nodes = set(np.flatnonzero(anc[cn_pop])) | set(np.flatnonzero(anc[:,cn_pop]))
        same_branch_deletions = deletions[doublet] & same_branch_nodes
        if len(same_branch_deletions) > 0:
          continue
      else:
        deletions[doublet] = set()
      deletions[doublet].add(cn_pop)

    triplets.add(triplet)
    if doublet not in directions:
      directions[doublet] = direction
    events.append(Cna(cn_pop, cn_seg, cn_phase, delta))

  return events

def _compute_allele_counts(struct, cna_events, H, ploidy):
  K = len(struct)
  root = 0
  alleles = np.nan * np.ones((K+1, H, ploidy))
  alleles[root,:,:] = 1

  # I can't use NaN in integer arrays, so use a silly value instead.
  parents = np.insert(struct, 0, -9999)

  _find_children = lambda P: np.flatnonzero(parents == P).tolist()
  stack = _find_children(root)
  while len(stack) > 0:
    pop = stack.pop()
    parent = parents[pop]
    alleles[pop] = alleles[parent]
    for event in cna_events:
      if event.pop != pop:
        continue
      assert event.delta != 0
      parent_cn = alleles[parent, event.seg, event.phase]
      alleles[pop, event.seg, event.phase] = parent_cn + event.delta
    stack += _find_children(pop)

  assert not np.any(np.isnan(alleles))
  assert np.all(alleles >= 0)
  return alleles

def generate_cnas(K, C, segs, struct, ploidy):
  H = len(segs)
  cna_events = _generate_cna_events(K, H, C, ploidy, struct)
  alleles = _compute_allele_counts(struct, cna_events, H, ploidy)
  return (cna_events, alleles)

def _compute_cna_influence(struct, cna_events, ssm_segs, ssm_pops, ssm_phases, ssm_timing):
  assert len(ssm_segs) == len(ssm_pops) == len(ssm_phases) == len(ssm_timing)
  M = len(ssm_segs)
  C = len(cna_events)

  # For `cna_influence`, we have an `MxC` matrix, where `cna_influence[i,j] =
  # 1` iff SSM `i` is influenced by CNA `j`. That is, SSM `i` occurred in the
  # same phase on the same segment as CNA `j` in an ancestral population to
  # where `j` occurred, or `i` occurred in the same phase on the same segment
  # as `j`  in the same population with timing such that `i` was before (not
  # after) `j`.
  infl = np.zeros((M, C), dtype=np.int8)
  adjm = _make_adjm(struct)
  anc = common.make_ancestral_from_adj(adjm)
  np.fill_diagonal(anc, 0)

  for cna_idx, event in enumerate(cna_events):
    anc_pops = np.flatnonzero(anc[event.pop])
    assert event.pop not in anc_pops
    ancestral_ssm_mask = np.logical_and.reduce((
      np.isin(ssm_pops, anc_pops),
      ssm_segs == event.seg,
      ssm_phases == event.phase,
    ))
    before_cna_ssm_mask = np.logical_and(
      ssm_pops == event.pop,
      ssm_timing == TIMING_BEFORE,
    )
    ssm_mask = np.logical_or(ancestral_ssm_mask, before_cna_ssm_mask)
    infl[ssm_mask, cna_idx] = 1

  return infl

def generate_ssms(K, M, G, S, T, segs, ploidy, phi, cna_events, alleles):
  # We ensure that every population has at least one SSM.
  ssm_pops = assign_ssms_to_pops(M, K) # Mx1
  clusters = make_clusters(ssm_pops)

  phase_probs = np.random.dirichlet(alpha = ploidy*[5])
  ssm_segs = []
  ssm_phases = []

  while len(ssm_segs) < M:
    # This could end up being an infinite loop -- we could have decided to
    # assign the SSM to a population where every segment in every phase is
    # deleted. Hopefully this won't ever happen, so I don't explicitly check
    # for it, but if a process is running forever, I should check this.
    ssmidx = len(ssm_segs)
    pop = ssm_pops[ssmidx]

    seg = np.random.choice(len(segs), p=segs)
    phase = np.random.choice(len(phase_probs), p=phase_probs)
    if alleles[pop,seg,phase] == 0:
      continue
    ssm_segs.append(seg)
    ssm_phases.append(phase)

  timing_probs = np.random.dirichlet(alpha = 2*[5])
  ssm_timing = np.random.choice(len(timing_probs), p=timing_probs, size=M)
  all_pops = set(range(1, K+1))
  assert set(ssm_pops) == all_pops
  cna_gain_pops = set([C.pop for C in cna_events if C.delta > 0])
  no_gain_pops = np.array(list(all_pops - cna_gain_pops))
  no_gain_ssms = np.isin(ssm_pops, no_gain_pops)
  ssm_timing[no_gain_ssms] = -1

  phi_good_mutations = np.array([phi[cidx] for cidx in ssm_pops]) # MxS
  phi_garbage = np.random.uniform(size=(G,S))
  phi_mutations = np.vstack((phi_good_mutations, phi_garbage))

  omega_v = np.broadcast_to(0.5, (M + G, S))
  variants = make_variants(phi_mutations, T, omega_v)
  vids_good = ['s%s' % vidx for vidx in range(M)]
  vids_garbage = ['s%s' % vidx for vidx in range(M, M + G)]
  assert set(vids_good) == set([V for C in clusters for V in C])

  return (
    variants,
    vids_good,
    vids_garbage,
    clusters,
    ssm_pops,
    np.array(ssm_segs),
    np.array(ssm_phases),
    ssm_timing,
  )

def convert_to_numpy_array(data):
  arrays = {K: np.array(data[K]) for K in (
    'structure',
    'segments',
    'phi',
    'eta',
    'ssm_pops',
    'ssm_segs',
    'ssm_phases',
    'ssm_timing',
    'cna_influence',
    'alleles',
    'seed',
  )}

  def _extract_attr(C, key):
    return np.array([getattr(c, key) for c in C])
  for key in ('pop', 'seg', 'phase', 'delta'):
    arrays['cna_%ss' % key] = _extract_attr(data['cna_events'], key)
  return arrays

def generate_data(K, S, T, M, C, H, G, alpha, tree_type):
  # K: number of clusters (excluding normal root)
  # S: number of samples
  # T: reads per mutation
  # M: total number of SSMs
  # C: total number of CNAs
  # H: number of genomic segments
  # G: number of (additional) garbage mutations
  ploidy = 2

  struct, phi, eta = generate_tree(K, S, alpha, tree_type)
  segs = segment_genome(H)
  cna_events, alleles = generate_cnas(K, C, segs, struct, ploidy)
  variants, \
    vids_good, \
    vids_garbage, \
    clusters, \
    ssm_pops, \
    ssm_segs, \
    ssm_phases, \
    ssm_timing = generate_ssms(K, M, G, S, T, segs, ploidy, phi, cna_events, alleles)
  cna_influence = _compute_cna_influence(struct, cna_events, ssm_segs, ssm_pops, ssm_phases, ssm_timing)

  return {
    'sampnames': ['Sample %s' % (sidx + 1) for sidx in range(S)],
    'structure': struct,
    'segments': segs,
    'phi': phi,
    'eta': eta,
    # TODO: remove clusters, since `ssm_pops` represents the same information
    # in a way more consistent with other variables (like `ssm_segs` and
    # `ssm_phases`)?
    'clusters': clusters,
    'variants': variants,
    'vids_good': vids_good,
    'vids_garbage': vids_garbage,
    'ssm_pops': ssm_pops,
    'ssm_segs': ssm_segs,
    'ssm_phases': ssm_phases,
    'ssm_timing': ssm_timing,
    'cna_events': cna_events,
    'cna_influence': cna_influence,
    'alleles': alleles,
  }
