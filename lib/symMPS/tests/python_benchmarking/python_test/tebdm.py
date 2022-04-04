import quimb.tensor as qtn
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from functools import reduce
from tqdm import tqdm


triplet = np.array(
[
    [1, 0, 0, 0],
    [0, 1 / 2, 1 / 2, 0],
    [0, 1 / 2, 1 / 2, 0],
    [0, 0, 0, 1],
]
)
singlet = np.eye(4)-triplet

def U(θ):
    return triplet + np.exp(-1j*θ)*singlet

class TEBDm(qtn.tensor_1d_tebd.TEBD):
    """TEBD for random circuit models. 
    This might be better expressed using arbitrary tensor networks, but for now this.  
    Although -- the measurements don't make sense in that picture. 
    """
    def __init__(self, p_u, p_m, *args, ancillae=0, add_ancillae=True, measure=True, **kwargs):
        """Create a measurement circuit

        Args:
            p: The measurement probability
            args: The args to pass to the TEBD object
            ancillae: The number of ancillae K. Will be entangled with the first K qubits of the state. 
                      by putting each ancillae into the |+> state and projecting pairs onto |00> + |11>.
            kwargs: kwargs to pass to the TEBD object. 
        """
        self.p_m = p_m
        self.p_u = p_u
        super().__init__(*args, **kwargs)
        self.L = len(self.pt.arrays)


    def _step_order1(self, tau=1, **sweep_opts):
        """Perform a single, first order step."""
        if not np.allclose(self.p_m, 0):
            self.sweep_measure("right", tau, **sweep_opts)
            self.sweep_measure("left", tau, **sweep_opts)
        if not np.allclose(self.p_u, 0):
            self.sweep_unitary("right", tau, **sweep_opts)
            self.sweep_unitary("left", tau, **sweep_opts)

    def step(self, order=1, dt=None, progbar=None, **sweep_opts):
        """Perform a single step of time ``self.dt``."""
        dt = 1

        self._step_order1(dt=dt, **sweep_opts)

        self.t += dt
        self._err += 0

        if progbar is not None:
            progbar.cupdate(self.t)
            self._set_progbar_desc(progbar)

    def at_times(self, ts, *args, **kwargs):
        return super().at_times(ts, *args, **kwargs)

    def update_to(self, T, *args, **kwargs):
        return super().update_to(T, *args, **kwargs)

    def sweep_measure(self, direction, dt_frac, dt=None, queue=False):
        """Perform a single sweep of measurements and compression. This shifts the
        orthogonality centre along with the gates as they are applied and
        split.

        Parameters
        ----------
        direction : {'right', 'left'}
            Which direction to sweep. Right is even bonds, left is odd.
        dt_frac : float
            What fraction of dt substep to take.
        dt : float, optional
            Overide the current ``dt`` with a custom value.
        """

        # if custom dt set, scale the dt fraction
        if dt is not None:
            dt_frac *= dt / self._dt

        dt_frac = 1

        if direction == "right":
            start_site_ind = 0
            final_site_ind = self.L - 1
            # Apply even gates:
            #
            #     o-<-<-<-<-<-<-<-<-<-   -<-<
            #     | | | | | | | | | |     | |       >~>~>~>~>~>~>~>~>~>~>~o
            #     UUU UUU UUU UUU UUU ... UUU  -->  | | | | | | | | | | | |
            #     | | | | | | | | | |     | |
            #      1   2   3   4   5  ==>
            #
            for i in range(start_site_ind, final_site_ind, 2):
                sites = (i, (i + 1) % self.L)
                # U = self._get_gate_from_ham(-10, sites)
                if np.random.rand() < self.p_m:
                    p_singlet = np.abs(self.pt.gate(singlet, sites).H@self.pt)
                    if np.random.rand() < p_singlet:
                        U = singlet
                    else:
                        U = triplet
                else:
                    U = np.eye(4)

                self._pt.left_canonize(start=max(0, i - 1), stop=i)
                self._pt.gate_split_(U, where=sites, absorb="right", **self.split_opts)

                # Renormalise after imaginary time evolution
                factor = np.sqrt(self.pt.H@self.pt)
                self._pt[final_site_ind] /= factor

            # odd
            if self.L % 2 == 1:
                self._pt.left_canonize_site(self.L - 2)
                if self.cyclic:
                    sites = (self.L - 1, 0)
                    # U = self._get_gate_from_ham(dt_frac, sites)
                    if np.random.rand() < self.p_m:
                        p_singlet = np.abs(self.pt.gate(singlet, sites).H@self.pt)
                        if np.random.rand() < p_singlet:
                            U = singlet
                        else:
                            U = triplet
                    else:
                        U = np.eye(4)
                    self._pt.right_canonize_site(1)
                    self._pt.gate_split_(
                        U, where=sites, absorb="left", **self.split_opts
                    )
                    # Renormalise after imaginary time evolution
                    factor = np.sqrt(self.pt.H@self.pt)
                    self._pt[final_site_ind] /= factor

        elif direction == "left":
            # PBC
            if self.cyclic and (self.L % 2 == 0):
                sites = (self.L - 1, 0)
                if np.random.rand() < self.p_m:
                    p_singlet = np.abs(self.pt.gate(singlet, sites).H@self.pt)
                    if np.random.rand() < p_singlet:
                        U = singlet
                    else:
                        U = triplet
                else:
                    U = np.eye(4)

                self._pt.right_canonize_site(1)
                self._pt.gate_split_(U, where=sites, absorb="left", **self.split_opts)

                # Renormalise after imaginary time evolution
                factor = np.sqrt(self.pt.H@self.pt)
                self._pt[final_site_ind] /= factor

            final_site_ind = 1
            # Apply odd gates:
            #
            #     >->->-   ->->->->->->->->-o
            #     | | |     | | | | | | | | |       o~<~<~<~<~<~<~<~<~<~<~<
            #     | UUU ... UUU UUU UUU UUU |  -->  | | | | | | | | | | | |
            #     | | |     | | | | | | | | |
            #           <==  4   3   2   1
            #
            for i in reversed(range(final_site_ind, self.L - 1, 2)):
                sites = (i, (i + 1) % self.L)


                if np.random.rand() < self.p_m:
                    p_singlet = np.abs(self.pt.gate(singlet, sites).H@self.pt)
                    if np.random.rand() < p_singlet:
                        U = singlet
                    else:
                        U = triplet
                else:
                    U = np.eye(4)

                self._pt.right_canonize(start=min(self.L - 1, i + 2), stop=i + 1)
                self._pt.gate_split_(U, where=sites, absorb="left", **self.split_opts)

                # Renormalise after imaginary time evolution
                factor = np.sqrt(self.pt.H@self.pt)
                self._pt[final_site_ind] /= factor


            # one extra canonicalization not included in last split
            self._pt.right_canonize_site(1)

    def sweep_unitary(self, direction, dt_frac, dt=None, queue=False):
        """Perform a single sweep of unitary gates and compression. This shifts the
        orthonognality centre along with the gates as they are applied and
        split.

        Parameters
        ----------
        direction : {'right', 'left'}
            Which direction to sweep. Right is even bonds, left is odd.
        dt_frac : float
            What fraction of dt substep to take.
        dt : float, optional
            Overide the current ``dt`` with a custom value.
        """

        # if custom dt set, scale the dt fraction
        if dt is not None:
            dt_frac *= dt / self._dt

        dt_frac = 1

        if direction == "right":
            start_site_ind = 0
            final_site_ind = self.L - 1
            # Apply even gates:
            #
            #     o-<-<-<-<-<-<-<-<-<-   -<-<
            #     | | | | | | | | | |     | |       >~>~>~>~>~>~>~>~>~>~>~o
            #     UUU UUU UUU UUU UUU ... UUU  -->  | | | | | | | | | | | |
            #     | | | | | | | | | |     | |
            #      1   2   3   4   5  ==>
            #
            for i in range(start_site_ind, final_site_ind, 2):
                sites = (i, (i + 1) % self.L)
                # U = self._get_gate_from_ham(-10, sites)
                self._pt.left_canonize(start=max(0, i - 1), stop=i)
                if np.random.rand() < self.p_u:
                    self._pt.gate_split_(U(np.random.rand()*2*np.pi), where=sites, absorb="right", **self.split_opts)
                else:
                    self._pt.gate_split_(np.eye(4), where=sites, absorb="right", **self.split_opts)

            # odd
            if self.L % 2 == 1:
                self._pt.left_canonize_site(self.L - 2)
                if self.cyclic:
                    sites = (self.L - 1, 0)
                    # U = self._get_gate_from_ham(dt_frac, sites)
                    if np.random.rand() < self.p_u:
                        self._pt.gate_split_(U(np.random.rand()*2*np.pi), where=sites, absorb="right", **self.split_opts)
                    else:
                        self._pt.gate_split_(np.eye(4), where=sites, absorb="right", **self.split_opts)

        elif direction == "left":
            # PBC
            if self.cyclic and (self.L % 2 == 0):
                sites = (self.L - 1, 0)
                if np.random.rand() < self.p_u:
                    self._pt.gate_split_(U(np.random.rand()*2*np.pi), where=sites, absorb="right", **self.split_opts)
                else:
                    self._pt.gate_split_(np.eye(4), where=sites, absorb="right", **self.split_opts)

            final_site_ind = 1
            # Apply odd gates:
            #
            #     >->->-   ->->->->->->->->-o
            #     | | |     | | | | | | | | |       o~<~<~<~<~<~<~<~<~<~<~<
            #     | UUU ... UUU UUU UUU UUU |  -->  | | | | | | | | | | | |
            #     | | |     | | | | | | | | |
            #           <==  4   3   2   1
            #
            for i in reversed(range(final_site_ind, self.L - 1, 2)):
                sites = (i, (i + 1) % self.L)

                if np.random.rand() < self.p_u:
                    self._pt.gate_split_(U(np.random.rand()*2*np.pi), where=sites, absorb="right", **self.split_opts)
                else:
                    self._pt.gate_split_(np.eye(4), where=sites, absorb="right", **self.split_opts)

            # one extra canonicalization not included in last split
            self._pt.right_canonize_site(1)
