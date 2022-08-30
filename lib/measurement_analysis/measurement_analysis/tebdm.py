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

        self.L = self.pt._L
        self.summer = qtn.tensor_gen.MPS_product_state([np.array([1.0, 1.0])] * (self.L))


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

        if progbar is not None:
            progbar.cupdate(self.t)
            self._set_progbar_desc(progbar)

    def forced_measure(self, odd):
        if 2*int(self.t)+odd >= self.measurement_locations.shape[0]:
            return None

        for i in range(self.L):
            if self.measurement_locations[2*int(self.t)+odd, i]:
                outcome = self.measurement_locations[2*int(self.t)+odd, i]
                P0 = np.array([[1, 0], [0, 0]])
                P1 = np.array([[0, 0], [0, 1]])

                if np.isclose(outcome, 1):  # 1 corresponds to rand() < ev(P0)
                    self._pt.gate_(P0, i, contract=True)
                    self.norm.append(np.abs(self.summer.H @ self._pt))
                    if self.norm[-1]<=1e-16:
                        raise ValueError('incompatible')
                    self._pt[0] /= self.norm[-1]  # renormalise
                elif np.isclose(outcome, -1): # rand() > ev(P0)
                    self._pt.gate_(P1, i, contract=True)
                    self.norm.append(np.abs(self.summer.H @ self._pt))
                    if self.norm[-1]<=1e-16:
                        print(2*int(self.t)+odd, i)
                        print(self.measurement_locations[2*int(self.t)+odd, i])
                        print(self.measurement_locations)
                        print(self.biases)
                        raise ValueError('incompatible')
                    self._pt[0] /= self.norm[-1]  # renormalise

    def measure(self):
        p_m = self.p
        for i in range(self.L):
            Z = np.array([[1, 0], [0, -1]])
            if self.measurement_locations[int(self.t), i]:
                p_bloch = (
                    1 + self.pt.gate(Z, i, contract=True).H @ self.summer
                ) / 2
                if np.random.rand() < p_bloch:
                    o = np.outer(np.array([1, 0]), np.array([1, 0]))
                    self._pt.gate_(o, i, contract=True)
                    self.norm.append(self.summer.H @ self.pt)
                    self._pt[-1] /= self.norm[-1]  # renormalise
                else:
                    o = np.outer(np.array([0, 1]), np.array([0, 1]))
                    self._pt.gate_(o, i, contract=True)
                    self.norm.append(self.summer.H @ self.pt)
                    self._pt[-1] /= self.norm[-1]  # renormalise

    def at_times(self, ts, measurement_locations=None, biases=None, *args, **kwargs):
        if measurement_locations is not None:
            self.forced=True
        else:
            self.forced=False

        self.measurement_locations = measurement_locations if measurement_locations is not None else np.random.choice(
            [0, 1], size=(2*len(ts), self.L), p=[1 - self.p, self.p]
        ) # This object is not stateful anymore - we'll run out of measurement record. 

        self.biases = biases if biases is not None else np.ones((self.L//2, 2*len(ts)+1))/2
        return super().at_times(ts, *args, **kwargs)

    def update_to(self, T, *args, **kwargs):
        return super().update_to(T, *args, **kwargs)

    def sweep(self, direction, dt_frac=1, dt=1, queue=False):
        """Perform a single sweep of gates and compression. This shifts the
        orthonognality centre along with the gates as they are applied and
        split.

        Parameters
        ----------
        direction : {'right', 'left'}
            Which direction to sweep. Right is even bonds, left is odd.
        """
        if 2*int(self.t)+1 >= self.biases.shape[1]:
            return None

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
                bias = self.biases[i//2, 2*int(self.t)]
                U = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1-bias, bias, 0],
                        [0, bias, 1-bias, 0],
                        [0, 0, 0, 1],
                    ]
                )

                self._pt.left_canonize(start=max(0, i - 1), stop=i)
                self._pt.gate_split_(U, where=sites, absorb="right", **self.split_opts)

            if self.L % 2 == 1:
                self._pt.left_canonize_site(self.L - 2)

        elif direction == "left":
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
                bias = self.biases[i//2+1, 2*int(self.t)+1]

                U = np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1-bias, bias, 0],
                        [0, bias, 1-bias, 0],
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
