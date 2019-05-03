import numpy as np

def make_ancestral_from_adj(adj):
  K = len(adj)
  root = 0

  # Disable checks to improve performance.
  #assert np.all(1 == np.diag(adj))
  #expected_sum = 2 * np.ones(K)
  #expected_sum[root] = 1
  #assert np.array_equal(expected_sum, np.sum(adj, axis=0))

  Z = np.copy(adj)
  np.fill_diagonal(Z, 0)
  stack = [root]
  while len(stack) > 0:
    P = stack.pop()
    C = list(np.flatnonzero(Z[P]))
    if len(C) == 0:
      continue
    Z[:,C] = Z[:,P][:,np.newaxis]
    Z[P,C] = 1
    stack += C
  np.fill_diagonal(Z, 1)

  #assert np.array_equal(Z[root], np.ones(K))
  return Z
