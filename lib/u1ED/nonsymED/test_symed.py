from symed import *
import numpy as np
from nonsymED import nonsymed
import time

def test_state():
    N = 10 
    Q = 0
    psi = nonsymed.state(N, N//2+Q)
    psi_ = state(N, Q).to_ndarray().reshape(-1)
    assert np.allclose(psi, psi_/np.linalg.norm(psi_))

def test_random_u1_unitary():
    U = nonsymed.random_u1_unitary()
    U_ = random_u1_unitary(U)
    assert np.allclose(U_.dot(U_.conj().transpose(), [1, 0]).to_ndarray(), np.eye(4))

def test_apply_two_site_op():
    N, Q = 24, 0
    site = 0

    t0 = time.time()
    psi_ = nonsymed.state(N, N//2+Q)
    U_ = nonsymed.random_u1_unitary()
    out_ = nonsymed.apply_two_site_op(psi_, site, U_)
    t1 = time.time()
    print(t1-t0)

    #t0 = time.time()
    #psi = state(N, Q)
    #U = random_u1_unitary(U_)
    #out = apply_two_site_op(psi, site, U)
    #t1 = time.time()
    #print(t1-t0)
    #assert np.allclose(out_, out/np.linalg.norm(out))
    raise Exception
    assert np.allclose(psi_.conj().T@psi_, 1)

if __name__=='__main__':
    test_state()
    test_random_u1_unitary()
    test_apply_two_site_op()

