import numpy as np
import sys


def popstep(P, M):
    return np.matmul(P, M)


def probInfection(baseSusceptibility, infected):
    return baseSusceptibility * infected


S1 = 0
S2 = 1
INF = 2
R = 3
NUM_STATES = 4

# Default params
S1_SUSCEPTIBILITY = 0.1
S2_SUSCEPTIBILITY = 1.0

INITIAL_S1_PROPORTION = 0.0
INITIAL_S2_PROPORTION = 1.0 - INITIAL_S1_PROPORTION
INITIAL_SEEDING = 1e-4
# Odds on a given day of recovering
I_R_PROBABILITY = 0.1
NUM_GENS = 100


def model_params(overrides):
    s1 = overrides.get("init_s1", INITIAL_S1_PROPORTION)
    s2 = overrides.get("init_s2", 1.0 - s1)
    s1_suscept = overrides.get("s1_suscept", S1_SUSCEPTIBILITY)
    s2_suscept = overrides.get("s2_suscept", S2_SUSCEPTIBILITY)
    seeding = overrides.get("seeding", INITIAL_SEEDING)
    recov = overrides.get('recovery', I_R_PROBABILITY)
    return {
      'init_s1': s1,
      'init_s2': s2,
      's1_suscept': s1_suscept,
      's2_suscept': s2_suscept,
      'seeding': seeding,
      'recovery_rate': recov,
    }


def generation(pop, M, params):
    pop = pop / np.sum(pop)
    # Recompute transition matrix
    M[S1][INF] = probInfection(params['s1_suscept'], pop[INF])
    M[S2][INF] = probInfection(params['s2_suscept'], pop[INF])
    # Row-wise normalize M
    for r in range(0, NUM_STATES):
        M[r] = M[r] / np.sum(M[r])
    return np.matmul(pop, M)


def generations(pop, M, params, num_gens):
    gens = np.zeros(shape=(NUM_GENS, NUM_STATES))
    for g in range(0, num_gens):
        pop = generation(pop, M, params)
        gens[g] = pop
    return gens


def init_pop(params):
    pop = np.zeros(shape=(NUM_STATES))
    pop[S1] = params['init_s1']
    pop[S2] = params['init_s2']
    pop[INF] = params['seeding']


def init_matrix(params):
    M = np.zeros(shape=(NUM_STATES, NUM_STATES))
    M[S1][S1] = 1.0
    M[S2][S2] = 1.0
    M[INF][R] = params['recovery_rate']
    M[INF][INF] = 1.0 - params['recovery_rate']
    M[R][R] = 1.0
    return M


def run(params, num_gens):
    pop = init_pop(params)
    M = init_matrix(params)
    return generations(pop, M, params, num_gens)


if __name__ == "__main__":
    gens = run(model_params({}), 100)
    np.save('generations.numpy', gens)
    sys.exit()
