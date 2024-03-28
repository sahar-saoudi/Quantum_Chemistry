import numpy as np
from qiskit.quantum_info import Pauli, PauliList, pauli_basis
from typing import Tuple
from numpy.typing import NDArray
from qiskit.providers import Backend
from qiskit import QuantumCircuit, QuantumRegister, transpile


def bitstring_to_bits(bit_string: str) -> NDArray[np.bool_]:
    """
    Converts a string of a bit into a numpy array of boolean values (the numpy array is reversed)

    Args: 
        bit_string: a string of "1" or "0" that represents a bit

    Returns:
        numpy array of boolean values representing the bit_string inversed

    """

    bits_list = np.array(list(bit_string))
    bits = np.flip(bits_list) == '1'

    return bits

def diag_pauli_expectation_value(pauli: Pauli, counts: dict) -> float:
    """
    Finds the expectation value of a diagonalized Pauli

    Args: 
        Pauli: diagonalized Pauli
        dict: Counts of the execution of the diagonalisation circuit


    Returns:
        float: returns the expectation of a diagonalized Pauli 

    """

    assert(np.all(~pauli.x))

    shots = sum(counts.values())
    pauli_z = pauli.z

    expectation_value = 0
    for bit_str ,value in counts.items():
        probabilitie = (float(value)/float(shots))
        bit_vector = bitstring_to_bits(bit_str).astype(int)
        valeur_propre = (1-2*(np.dot(pauli_z.astype(int), bit_vector)%2))
        expectation_value = expectation_value+probabilitie*valeur_propre
        
    return expectation_value



def diagonalize_pauli_with_circuit(pauli : Pauli) -> Tuple[Pauli, QuantumCircuit]:
    """
    diagonalize a pauli with a circuit to execute

    Args: 
        Pauli: pauli to diagonalize

    Returns:
        Pauli: Diagonalized pauli
        QuantumCircuit: Circuit to execute for the diagonalisation of the pauli
    """

    number_of_qubits = pauli.num_qubits
    
    qreg = QuantumRegister(number_of_qubits, "q")
    circuit = QuantumCircuit(qreg)
    diag_pauli_x = list(np.full(len(pauli.x),False))
    diag_pauli_z = np.logical_or(pauli.z, pauli.x)

    for i in range(number_of_qubits):
        if pauli.x[i]:
            if pauli.z[i]:
                circuit.sdg(qreg[i])

            circuit.h(qreg[i])  
            
       
                
    diagonal_pauli = Pauli((diag_pauli_z, diag_pauli_x))
    assert(np.all(~diagonal_pauli.x))

    return diagonal_pauli, circuit

def estimate_expectation_values(paulis: PauliList, state_circuit: QuantumCircuit, backend: Backend, execute_opts : dict = dict()) -> NDArray[np.float_]:
    """
    estimate the expectation values of a state circuit with all the paulis possible with the number of qubits for the state circuit

    Args: 
        PaulisList: list of the paulis to use to find the expectation values
        state_circuit: circuit of the quantum state
        backend: backend on which we want to execute the circuit
        execute_opts: options for the execution of the circuit

    Returns:
        numpy array containing all the expectation values
    """

    number_of_qubits = state_circuit.num_qubits
    diag_paulis = []
    circuits = []

    for pauli in paulis:
        diag_pauli, diag_circuit = diagonalize_pauli_with_circuit(pauli)
        diag_paulis.append(diag_pauli)
        qreg = QuantumRegister(number_of_qubits, "q")
        current_circuit = QuantumCircuit(qreg)
        current_circuit.append(state_circuit.to_gate(), qreg)
        current_circuit.append(diag_circuit.to_gate(), qreg)
        current_circuit.measure_all()
        circuits.append(current_circuit)

    transpiled_circuits = transpile(circuits, backend)
    job = backend.run(transpiled_circuits, shots = execute_opts["shots"])
    counts = job.result().get_counts()

    expectation_values = np.empty(len(counts))
    for i, (current_pauli, current_counts) in enumerate(zip(diag_paulis, counts)):
        expectation_values[i] = diag_pauli_expectation_value(current_pauli, current_counts)


    return expectation_values


def state_tomography(state_circuit: QuantumCircuit, backend: Backend, execute_opts : dict = dict()) -> NDArray[np.complex_]:
    """
    finds the statevector of a quantum state using the circuit which constructs the state

    Args: 
        state_circuit: circuit of the quantum state
        backend: backend on which the program will execute the circuits
        execute_opts: options for the execution of the circuit

    Returns:
        a numpy array representing the statevector
    """

    nb_qb = state_circuit.num_qubits
    paulis = pauli_basis(nb_qb)

    coefficients = estimate_expectation_values(paulis, state_circuit, backend, execute_opts)/(2**nb_qb)

    paulis_matrices=paulis.to_matrix(array=True)
    dens_mat=np.tensordot(coefficients,paulis_matrices,axes=1)

    evals, evecs = np.linalg.eig(dens_mat)
    statevector = evecs[:, np.argmax(evals)]

    return statevector
