import numpy as np
import sys

def popstep(P, M):
    return np.matmul(P, M)


def probInfection(baseSusceptibility, infected):
    return baseSusceptibility * infected

S1=0
S2=1
I=2
R=3
NUM_STATES=4

S1_SUSCEPTIBILITY=1.0
S2_SUSCEPTIBILITY=0.1

INITIAL_S1_PROPORTION=0.3
INITIAL_S2_PROPORTION=1.0 - INITIAL_S1_PROPORTION
INITIAL_SEEDING=1e-4
# Odds on a given day of recovering
I_R_PROBABILITY=0.1
NUM_GENS=100

if __name__ == "__main__":
    print("oy")
    # import pdb; pdb.set_trace()

    pop = np.zeros(shape=(NUM_STATES))
    
    M = np.zeros(shape=(NUM_STATES,NUM_STATES))
    gens = np.zeros(shape=(NUM_GENS, NUM_STATES))
    pop[S1] = INITIAL_S1_PROPORTION
    pop[S2] = INITIAL_S2_PROPORTION
    pop[I] = INITIAL_SEEDING

    M[S1][S1] = 1.0
    M[S2][S2] = 1.0
    M[I][R] = I_R_PROBABILITY
    M[I][I] = 1.0 - I_R_PROBABILITY
    M[R][R] = 1.0

    peakI = 0.0
    for gen in range(0, 100):
        pop = pop / np.sum(pop)
        gens[gen] = pop
        # Recompute transition matrix
        M[S1][I] = probInfection(S1_SUSCEPTIBILITY, pop[I])
        M[S2][I] = probInfection(S2_SUSCEPTIBILITY, pop[I])
        # Row-wise normalize M
        for r in range(0, NUM_STATES):
            M[r] = M[r] / np.sum(M[r])
        # print(gen, M, pop)
        if pop[I] > peakI:
            print("New peak: gen {} {}".format(gen, pop[I]))
            peakI = pop[I]
        pop = popstep(pop, M)

    np.save('generations.numpy', gens) 
    sys.exit()
