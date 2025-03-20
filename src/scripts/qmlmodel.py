import qiskit 
import qiskit.quantum_info as qiskit_quantum_info
import numpy as np

from scipy.special import softmax

NBQUBITS=2

def make_circuit(x, t):
    """ x is normalized, x is an angle"""
    qc = qiskit.QuantumCircuit(NBQUBITS)
    qc.ry(x[0],0,'x0'), qc.ry(x[1],1,'x1'), qc.cz(0,1)
    qc.ry(t[0],0,'t0'), qc.ry(t[1],1,'t1'), qc.cz(0,1)
    qc.ry(x[2],0,'x2'), qc.ry(x[3],1,'x3'), qc.cz(0,1)
    qc.ry(t[2],0,'t2'), qc.ry(t[3],1,'t3')
    return qc

def rescale_to_angle(X, X_all):
    X_min, X_max = np.min(X_all, axis=0), np.max(X_all, axis=0)
    return np.pi * (X - X_min) / (X_max - X_min) + np.pi/2

def predict(x,t,nbshots=0):
    return predict_proba(x, t, nbshots).argmax(axis=1)

def predict_proba(X, t, nbshots=0):
    proba = np.array([_predict_proba(x,t) for x in X])  
    return proba  

def _predict_proba(x, t, nbshots=0):
    qc = make_circuit(x, t)
    stateVec = qiskit_quantum_info.Statevector.from_instruction(qc)
    probavec = stateVec.probabilities()    
    if not nbshots: return probavec[:-1]
    shots = np.random.multinomial(nbshots, probavec)
    return shots[:-1]/nbshots

def loss(t,X_train,y_true, nbshots=0):
    y_pred_proba = predict_proba(X_train, t, nbshots)
    y_pred_proba = softmax(y_pred_proba,axis=1)
    return np.mean(-np.log(y_pred_proba[np.arange(len(y_true)), y_true]))