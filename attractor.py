import numpy as np
import matplotlib.pyplot as plt

class Attractor():
    """Attracts the moving point."""
    def __init__(self, pos, force, exp_w=-2):
        """
        Parameters
        ----------
        pos : ndarray
            Position of the attractor.
        force : float
            Strenght of the attractor.
        exp_w : float
            Dependency of the distance. The default is -2 (gravity-like).
        """

        self.pos = pos
        self.f = force
        self.exp_w = exp_w

    def __call__(self, point):
        """Determins the direction and magnitude of the force applied to the point."""
        r = np.linalg.norm(self.pos-point, ord=2)
        direction = r**self.exp_w * self.f * (self.pos - point)
        return direction

    def __str__(self):
        plt.scatter(self.pos[1], self.pos[0], linewidths=self.f, marker='o')
        return ''

class System():
    """Models a system of several attractors and a moving point."""
    def __init__(self, point, num_attractors, v_init=None):
        """
        Parameters
        ----------
        point : ndarray
            Initial position of the moving point.
        num_attractors : int
            Number of attractors.
        v_init : ndarray, optional
            Initial velocity. The default is None.
        """

        # Create the moving point
        self.p = point
        if v_init is None:
            self.v = np.array([0, 0]).astype('float64')
        else:
            self.v = v_init

        # Create the attractors
        self.a = list()
        for i in range(num_attractors):
            self.a.append(Attractor([np.random.uniform(-5, 5), np.random.uniform(-5, 5)], np.random.uniform(0.5, 1.5)))

    def iterate(self, step_size, total_steps):
        """
        Iteratively update the velocity and the position of the moving point.

        Parameters
        ----------
        step_size : float
            Size of the step in the iteration.
        total_steps : int
            Total number of steps.

        Returns
        -------
        poses : ndarray
            Array of all positions in time.

        """

        poses = np.zeros((total_steps,2))
        for i in range(total_steps):
            for attractor in self.a:
                self.v += attractor(self.p)
            self.p = self.p + step_size * self.v
            poses[i, :] = self.p
        return poses

    def __str__(self):
        for a in self.a:
            print(a)
        return ''


if __name__ == "__main__":
    # System initialization
    p = np.array([0, 0])
    n_a = 4
    v_init = np.array([1000, 0]).astype('float64')
    s = System(p, n_a, v_init)

    # Create the path
    path = s.iterate(0.000006, 35000)

    # Plot the attractors and the path of the moving point.
    print(s)
    plt.plot(path[:, 0], path[:, 1], linewidth=0.5, alpha=1)
    plt.show()