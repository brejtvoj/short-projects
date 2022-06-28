import numpy as np
import matplotlib.pyplot as plt


def generate_points(num, interval=10):
    """Generate num points to which to travel."""
    return np.random.uniform(-interval, interval, (2, num))

def dist(p1, p2):
    """Distance between two points."""
    return np.linalg.norm(p1-p2, ord=2)

def all_dists(points, exp_w):
    """Calculate the distance between each pair of points"""
    eps = 1e-31
    dists = np.subtract.outer(points, points)
    dists = np.sqrt(dists[0, :, 0, :]**2 + dists[1, :, 1, :]**2)
    return (dists + eps)**exp_w

def choose_path(dists, weights, visited, start):
    """Choose which path to take based on the distance, whether the point was already visited
    and the weight."""
    dists_w = weights[start, :] * dists[start, :] # Combine found weights and distances
    dists_w_v = dists_w * (1 - visited) # Invalidate visited points
    d_f =  dists_w_v.flatten()
    prob_visit = d_f / np.sum(d_f) # Create PDF
    chosen_idx = np.random.choice(np.arange(dists.shape[0]), p=prob_visit)
    return chosen_idx, dists_w_v[chosen_idx]

def path_cost(dists, moves):
    """Calculate the total distance travelled."""
    cost = 0
    for m in moves:
        cost += dists[m[0], m[1]]
    return cost

def find_path(dists, weights):
    """Move through all points."""
    # Initialization
    start = np.random.choice(np.arange(dists.shape[0]))
    visited = np.zeros(dists.shape[0])
    visited[start] = 1
    cost = 0
    moves = list()

    # Move
    for i in range(dists.shape[0] - 1):
        idx, cost_i = choose_path(dists, weights, visited, start)
        move_i = (start, idx)
        moves.append(move_i)
        start = idx
        cost += cost_i
        visited[idx] = 1.0
    return moves, cost

def update_weight(weights, moves, cost, best_cost, it, it_total):
    """Change weight based on if the found path is better or worse then the previous best."""
    def sigmoid():
        return 1 / (1 + np.exp(-cost + best_cost))

    for m in moves:
        weights[m[0], m[1]] += 2. * sigmoid() - 1 # <-1, 1> range (0) if no change
    return np.clip(weights, 0.1, np.inf) # Limit the low-bound of weights

def travel(points, n_iters=1000, exp_w=-1, visualize=True):
    """Move through all the points n_iter times, update the weights of different roads
        and find the shortest one."""
    # Initilization
    d = all_dists(points, exp_w)
    w = np.ones_like(d)
    best_cost = np.inf
    best_move = None
    d_c = all_dists(points, 1)
    
    if visualize:
        plot_net(points, w)

    for i in range(n_iters):
        m_i, _ = find_path(d, w)
        c_i = path_cost(d_c, m_i)
        
        w = update_weight(w, m_i, c_i, best_cost, i, n_iters)

        # Update the best
        if c_i < best_cost:
            best_cost = c_i
            best_move = m_i
            if visualize:
                plot_net(points, w)

    return best_move, best_cost, w



# Visualization
def plot_net(p, w):
    w_c = w.copy()
    w_c = w_c / np.sum(w_c, axis=0)
    plt.scatter(p[0], p[1])
    for i in range(n_points):
        for j in range(n_points):
            start = p[:, i]
            end = p[:, j]
            plt.plot([start[0], end[0]], [start[1], end[1]], c='red', alpha=w_c[i, j])
    plt.show()
    
def plot_best(p, m):
    plt.scatter(p[0], p[1])
    for m_i in m:
        start = p[:, m_i[0]]
        end = p[:, m_i[1]]
        plt.plot([start[0], end[0]], [start[1], end[1]], c='red')
    plt.show()


np.random.seed(122460897)

if __name__ == "__main__":
    n_points = 16
    n_iters = 10000

    p = generate_points(n_points)

    m, cost, w = travel(p, n_iters)
    plot_best(p, m)
    plot_net(p, w)
    
