import quimb.tensor as qtn
import numpy as np


class TEBDm(qtn.tensor_1d_tebd.TEBD):
    """TEBD for random circuit models. 
    This might be better expressed using arbitrary tensor networks, but for now this.  
    """
    def __init__(self, p, *args, ancillae=0, add_ancillae=True, measure=True, **kwargs):
        """Create a measurement circuit

        Args:
            p: The measurement probability
            args: The args to pass to the TEBD object
            ancillae: The number of ancillae K. Will be entangled with the first K qubits of the state. 
                      by putting each ancillae into the |+> state and projecting pairs onto |00> + |11>.
            kwargs: kwargs to pass to the TEBD object. 
        """
        self.norm = []
        self.p = p
        self.should_measure=measure
        super().__init__(*args, **kwargs)

        if ancillae == 0 or add_ancillae:
            self.L = self.pt._L
        else:
            self.L = self.pt._L

        self.ancillae = ancillae
        self.summer = qtn.tensor_gen.MPS_product_state([np.array([1.0, 1.0])] * (self.L+self.ancillae))


    def _step_order1(self, tau=1, **sweep_opts):
        """Perform a single, first order step."""
        self.sweep("right", tau, **sweep_opts) # even sites
        if self.should_measure:
            if self.forced:
                self.forced_measure(0)
            else:
                self.measure()

        self.sweep("left", tau, **sweep_opts) # odd sites
        if self.should_measure:
            if self.forced:
                self.forced_measure(1)
            else:
                self.measure()

    def step(self, order=1, dt=None, progbar=None, **sweep_opts):
        """Perform a single step of time ``self.dt``."""
        dt = 1

        self._step_order1(dt=dt, **sweep_opts)

        self.t += dt
        self._err += self._ham_norm * dt ** (order + 1)

        if progbar is not None:
            progbar.cupdate(self.t)
            self._set_progbar_desc(progbar)

    def forced_measure(self, odd):
        #print(self.L)
        for i in range(self.L):
            outcome = self.measurement_locations[2*int(self.t)+odd, i]
            P0 = np.array([[1, 0], [0, 0]])
            P1 = np.array([[0, 0], [0, 1]])
            if not np.isclose(outcome, 0):
                if np.isclose(outcome, 1):  # 1 corresponds to rand() < ev(P0)
                    self._pt.gate_(P0, i, contract=True)
                    self.norm.append(self.summer.H @ self._pt)
                    if np.isclose(self.norm[-1], 0):
                        raise Exception('incompatible')

                    self._pt[0] /= self.norm[-1]  # renormalise
                else: # rand() > ev(P0)
                    self._pt.gate_(P1, i, contract=True)
                    self.norm.append(self.summer.H @ self._pt)
                    if np.isclose(self.norm[-1], 0):
                        raise Exception('incompatible')

                    self._pt[0] /= self.norm[-1]  # renormalise

    def measure(self):
        p_m = self.p
        for i in range(self.L):
            Z = np.array([[1, 0], [0, -1]])
            if self.measurement_locations[int(self.t), i]:
                p_bloch = (
                    1 + self.pt.gate(Z, i, contract=True).H @ self.summer
                ) / 2  # = p^2 - (1-p)^2 = 2*p-1
                if np.random.rand() < p_bloch:
                    o = np.outer(np.array([1, 0]), np.array([1, 0]))
                    self._pt.gate_(o, i, contract=True)
                    # self._pt.measure(i, outcome=0, renorm=False, inplace=True)
                    self.norm.append(self.summer.H @ self.pt)
                    self._pt[-1] /= self.norm[-1]  # renormalise
                else:
                    o = np.outer(np.array([0, 1]), np.array([0, 1]))
                    self._pt.gate_(o, i, contract=True)
                    # self._pt.measure(i, outcome=1, renorm=False, inplace=True)
                    self.norm.append(self.summer.H @ self.pt)
                    self._pt[-1] /= self.norm[-1]  # renormalise

    def at_times(self, ts, measurement_locations=None, *args, **kwargs):
        self.measurement_locations = measurement_locations if measurement_locations is not None else np.random.choice(
            [0, 1], size=(self.L, len(ts)), p=[1 - self.p, self.p]
        ).T
        if self.measurement_locations is not None:
            self.forced=True
        return super().at_times(ts, *args, **kwargs)

    def update_to(self, T, *args, **kwargs):
        #self.measurement_locations = np.random.choice(
        #    [0, 1], size=(self.L, int(T)), p=[1 - self.p, self.p]
        #).T
        return super().update_to(T, *args, **kwargs)

    def sweep(self, direction, dt_frac, dt=None, queue=False):
        """Perform a single sweep of gates and compression. This shifts the
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
                dt_frac = 1
                # U = self._get_gate_from_ham(-10, sites)
                U = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1 / 2, 1 / 2, 0],
                        [0, 1 / 2, 1 / 2, 0],
                        [0, 0, 0, 1],
                    ]
                )

                self._pt.left_canonize(start=max(0, i - 1), stop=i)
                self._pt.gate_split_(U, where=sites, absorb="right", **self.split_opts)

            if self.L % 2 == 1:
                self._pt.left_canonize_site(self.L - 2)
                if self.cyclic:
                    sites = (self.L - 1, 0)
                    # U = self._get_gate_from_ham(dt_frac, sites)
                    U = np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 1 / 2, 1 / 2, 0],
                            [0, 1 / 2, 1 / 2, 0],
                            [0, 0, 0, 1],
                        ]
                    )
                    self._pt.right_canonize_site(1)
                    self._pt.gate_split_(
                        U, where=sites, absorb="left", **self.split_opts
                    )
        elif direction == "left":
            if self.cyclic and (self.L % 2 == 0):
                sites = (self.L - 1, 0)
                # U = self._get_gate_from_ham(dt_frac, sites)
                U = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1 / 2, 1 / 2, 0],
                        [0, 1 / 2, 1 / 2, 0],
                        [0, 0, 0, 1],
                    ]
                )
                self._pt.right_canonize_site(1)
                self._pt.gate_split_(U, where=sites, absorb="left", **self.split_opts)

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
                # U = self._get_gate_from_ham(dt_frac, sites)
                U = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1 / 2, 1 / 2, 0],
                        [0, 1 / 2, 1 / 2, 0],
                        [0, 0, 0, 1],
                    ]
                )
                self._pt.right_canonize(start=min(self.L - 1, i + 2), stop=i + 1)
                self._pt.gate_split_(U, where=sites, absorb="left", **self.split_opts)

            # one extra canonicalization not included in last split
            self._pt.right_canonize_site(1)

        # Renormalise after imaginary time evolution
        factor = self.summer.H @ self.pt
        self._pt[-1] /= factor

