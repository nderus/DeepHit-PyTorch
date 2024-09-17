import numpy as np
import torch
from lifelines import KaplanMeierFitter

### C(t)-INDEX CALCULATION
def c_index(Prediction, Time_survival, Death, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other causes)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    N = len(Prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    Num = 0
    Den = 0

    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1

        if (Time_survival[i] <= Time and Death[i] == 1):
            N_t[i, :] = 1

    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)

    if Num == 0 and Den == 0:
        result = -1  # Not able to compute c-index!
    else:
        result = float(Num / Den)

    return result

### BRIER-SCORE
def brier_score(Prediction, Time_survival, Death, Time):
    N = len(Prediction)
    y_true = ((Time_survival <= Time) * Death).astype(float)
    return np.mean((Prediction - y_true) ** 2)

##### WEIGHTED C-INDEX & BRIER-SCORE
def CensoringProb(Y, T):
    T = T.reshape([-1])  # (N,) - np array
    Y = Y.reshape([-1])  # (N,) - np array

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(Y == 0).astype(int))  # Censoring prob = survival probability of event "censoring"
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()

    # Fill 0 with ZoH (to prevent NaN values)
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]
    
    return G

### C(t)-INDEX CALCULATION: this accounts for the weighted average for unbiased estimation
def weighted_c_index(T_train, Y_train, Prediction, T_test, Y_test, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    G = CensoringProb(Y_train, T_train)

    N = len(Prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    Num = 0
    Den = 0

    for i in range(N):
        tmp_idx = np.where(G[0, :] >= T_test[i])[0]

        if len(tmp_idx) == 0:
            W = (1. / G[1, -1]) ** 2
        else:
            W = (1. / G[1, tmp_idx[0]]) ** 2

        A[i, np.where(T_test[i] < T_test)] = 1. * W
        Q[i, np.where(Prediction[i] > Prediction)] = 1.  # Give weights

        if (T_test[i] <= Time and Y_test[i] == 1):
            N_t[i, :] = 1.

    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)

    if Num == 0 and Den == 0:
        result = -1  # Not able to compute c-index!
    else:
        result = float(Num / Den)

    return result

### WEIGHTED BRIER SCORE: This accounts for the weighted average for unbiased estimation
def weighted_brier_score(T_train, Y_train, Prediction, T_test, Y_test, Time):
    G = CensoringProb(Y_train, T_train)
    N = len(Prediction)

    W = np.zeros(len(Y_test))
    Y_tilde = (T_test > Time).astype(float)

    for i in range(N):
        tmp_idx1 = np.where(G[0, :] >= T_test[i])[0]
        tmp_idx2 = np.where(G[0, :] >= Time)[0]

        if len(tmp_idx1) == 0:
            G1 = G[1, -1]
        else:
            G1 = G[1, tmp_idx1[0]]

        if len(tmp_idx2) == 0:
            G2 = G[1, -1]
        else:
            G2 = G[1, tmp_idx2[0]]

        W[i] = (1. - Y_tilde[i]) * float(Y_test[i]) / G1 + Y_tilde[i] / G2

    y_true = ((T_test <= Time) * Y_test).astype(float)

    return np.mean(W * (Y_tilde - (1. - Prediction)) ** 2)