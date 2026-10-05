"""
transport_algebra.py — the one receipt behind the Note.

Pure numpy, random matrices, no language model. It checks ALGEBRA only:
what J-CoT's read-in is, mathematically, once the vocabulary labels are
written out. It says nothing about how large the effects are in a real
transformer. That is Gate 0 in README.md.

Setup (J-CoT, arXiv 2607.21981, eq. 4-7):
  J_l      averaged Jacobian  d h_final / d h_l          (d x d)
  u_v      final-layer direction of word v (row of W_U) (d)
  d_{l,v}  J-lens atom at layer l  = J_l^T u_v           ("speaking"/read covector)
  write layer w, read layer r.  J-CoT extracts a at w, writes D_r a at r.

Claims checked:
  1. Unnormalised read-in  D_r a  ==  T h_w,  T = J_r^T J_w^{-T}.
     The vocabulary index cancels: transport is one fixed linear map.
  2. T == S^{-T}, where S = J_r^{-1} J_w is the effect-preserving map
     (the r-layer change with the same first-order downstream effect as h_w).
  3. T == S exactly when S is orthogonal.
  4. Per axis of a channel, a read weight is the stretch, a write weight is
     one over the stretch: reading and writing agree only where stretch == 1.
  5. J-CoT normalises atoms per layer; then read-in == T h_w with a per-word
     gain n_{w,v}/n_{r,v}. Not exactly linear any more, but the direction
     problem is unchanged.
  6. Prop. 1's recovery defect eps_l(a) = ||Phi_l(Psi_l(a)) - a|| involves
     no model computation: it is a property of the dictionary alone.

Run:  python transport_algebra.py
"""
import numpy as np

rng = np.random.default_rng(0)
d, V, k = 64, 2000, 25


def jac(scale_cols=None):
    J = np.eye(d) + 0.3 * rng.standard_normal((d, d))
    if scale_cols is not None:
        J[:, :scale_cols[0]] *= scale_cols[1]   # squash a block: ill-conditioned
    return J


def cos(x, y):
    return float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))


U = rng.standard_normal((V, d))                 # word directions (rows of W_U)
a = np.zeros(V)
a[rng.choice(V, k, replace=False)] = rng.random(k)   # a sparse J-thought

print("=" * 66)
for label, Jr in [("generic random Jacobians (I + noise; NOT a real model)", jac()),
                  ("early read layer, collapsed (cf. Anthropic Fig.28d)", jac((48, 0.05)))]:
    Jw = jac()
    Dr, Dw = Jr.T @ U.T, Jw.T @ U.T              # columns are atoms
    hw = Dw @ a
    T = Jr.T @ np.linalg.inv(Jw.T)
    S = np.linalg.inv(Jr) @ Jw
    sv = np.linalg.svd(S, compute_uv=False)
    print(label)
    print("  [1] read-in D_r a == T h_w           :", np.allclose(Dr @ a, T @ hw))
    print("  [2] T == S^{-T}                      :", np.allclose(T, np.linalg.inv(S).T))
    print("  S singular values  max %.3g  min %.3g  (orthogonal iff all == 1)" % (sv.max(), sv.min()))
    print("  cos(T h_w, S h_w) on the J-thought   : %.3f" % cos(T @ hw, S @ hw))
    # [5] per-layer atom normalisation, as J-CoT does
    nr, nw = np.linalg.norm(Dr, axis=0), np.linalg.norm(Dw, axis=0)
    Drn, Dwn = Dr / nr, Dw / nw
    hw_n = Dwn @ a
    readin_n = Drn @ a
    print("  [5] normalised read-in == T h_w with gain n_w/n_r :",
          np.allclose(T @ hw_n, Drn @ ((nr / nw) * a)))
    print("      cos(normalised read-in, T h_w)  : %.3f" % cos(readin_n, T @ hw_n))
    print("-" * 66)

# [3] orthogonal S  ->  T == S
Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
Jw = jac()
Jr = Jw @ Q.T                                   # makes S = Jr^{-1} Jw = Q
T = Jr.T @ np.linalg.inv(Jw.T)
S = np.linalg.inv(Jr) @ Jw
print("[3] S orthogonal  ->  T == S          :", np.allclose(T, S))

# [4] one channel, diagonal stretch: read = stretch, write = 1/stretch
s = np.array([4.0, 1.0, 0.25])
J = np.diag(s)                                  # channel: output = J @ state
read = J.T @ np.eye(3)                          # pulled-back output directions
write = np.linalg.inv(J) @ np.eye(3)            # state change giving unit output change
print("[4] stretch          :", s)
print("    read weights     :", np.diag(read))
print("    write weights    :", np.diag(write), " (agree only where stretch == 1)")

# [6] Prop. 1's eps uses only the dictionary: reconstruct, re-extract, same layer
Jl = jac()
Dl = Jl.T @ U.T
Dl /= np.linalg.norm(Dl, axis=0)
h = Dl @ a
supp = np.flatnonzero(a)
a_hat = np.zeros(V)
a_hat[supp] = np.linalg.lstsq(Dl[:, supp], h, rcond=None)[0]   # oracle-support extraction
print("[6] eps_l(a) with oracle support      : %.2e  (no transformer block was run)"
      % np.linalg.norm(a_hat - a))
print("=" * 66)
