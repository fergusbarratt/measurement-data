from nonsymED.nonsymed import *
import numpy as np
np.random.seed(500)

def get_psi(L):
    psi = np.random.randn(2**L) + 1j*np.random.randn(2**L)
    psi /= np.linalg.norm(psi)
    return psi
    

def test_apply_random_unitaries():
    L = 10
    psi = get_psi(L)
    psi_ = apply_random_unitaries(psi, 0)
    assert np.allclose(psi.conj().T@psi, 1)
    assert np.allclose(psi_.conj().T@psi_, 1)

def test_apply_random_measurements_twice():
    # test two measurements get the same result
    L = 10
    psi = get_psi(L)
    psic = psi.copy()
    sites_mask = np.random.randint(0, 2, size=(L,))
    psi_ = apply_random_measurements(psi, sites_mask)
    old_mask = sites_mask.copy()
    psi__ = apply_random_measurements(psi_, sites_mask)

    assert np.allclose(old_mask, sites_mask) # no changes if measuring the same thing twice. 
    assert np.allclose(psi_, psi__)
    assert np.allclose(psi_.conj().T@psi_, 1)

def test_apply_random_measurements_no_mutate():
    # doesn't mutate psi
    L = 10
    psi = get_psi(L)
    psic = psi.copy()
    sites_mask = np.random.randint(0, 2, size=(L,))
    old_mask = sites_mask.copy()

    psi_ = apply_random_measurements(psi, sites_mask)

    assert not np.allclose(sites_mask, old_mask) # does mutate sites_mask
    assert not np.allclose(psi, psi_)
    assert np.allclose(psi, psic)


def test_apply_op():
    # doesn't mutate psi
    L = 10
    psi = get_psi(L)
    psic = psi.copy()
    psi_ = apply_op(psi, 0, np.random.randn(2, 2))
    assert not np.allclose(psi, psi_)
    assert np.allclose(psi, psic)

if __name__=='__main__':
    test_apply_op()
    test_apply_random_measurements_no_mutate()
    test_apply_random_measurements_twice()
    test_apply_random_unitaries()
