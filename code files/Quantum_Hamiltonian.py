from qiskit.quantum_info import SparsePauliOp, Pauli
from typing import List, Callable
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from scipy.optimize import OptimizeResult
from qiskit.providers import Backend
from tomography import estimate_expectation_values


def annihilation_operators_with_jordan_wigner(num_states: int) -> List[SparsePauliOp]:
    """
    Builds the annihilation operators as sum of two Pauli Strings for given number offermionic
    states using the Jordan Wigner mapping.

    Args:
    num_states (int): Number of fermionic states.
    Returns:
    List[SparsePauliOp]: The annihilation operators
    """
    annihilation_operators = []
    tri_array = np.tri(num_states+1, M=num_states, k=-1, dtype = np.bool_)
    identity = np.eye(num_states, dtype = np.bool_)

    for i in range(num_states):
        pauli_1 = Pauli((tri_array[i], identity[i]))
        pauli_2 = Pauli((tri_array[i+1], identity[i]))

        sp_pauli = SparsePauliOp([pauli_1, pauli_2], [1/2, complex(0, 1/2)])
        annihilation_operators.append(sp_pauli)

    return annihilation_operators




def build_qubit_hamiltonian(one_body: NDArray[np.complex_], two_body: NDArray[np.complex_], annihilation_operators: List[SparsePauliOp], creation_operators: List[SparsePauliOp]) -> SparsePauliOp:
    """
    Build a qubit Hamiltonian from the one body and two body fermionic Hamiltonians.
    Args:
    one_body (NDArray[np.complex_]): The matrix for the one body Hamiltonian
    two_body (NDArray[np.complex_]): The array for the two body Hamiltonian
    annihilation_operators (List[SparsePauliOp]): List of sums of two Pauli strings
    creation_operators (List[SparsePauliOp]): List of sums of two Pauli strings (adjoint of
    annihilation_operators)
    Returns:
    SparsePauliOp: The total Hamiltonian as a sum of Pauli strings

    """
    operators = []

    #one_body_hamiltonian
    for i in range(len(one_body)):
        for j in range(len(one_body)):
            hij = one_body[i, j]
            operators.append(creation_operators[i].compose(annihilation_operators[j])*hij)
            two_body_ij = creation_operators[i].compose(creation_operators[j])        
            #two_body_hamiltonian
            for k in range(len(two_body)):
                two_body_ijk = two_body_ij.compose(annihilation_operators[k])
                for l in range(len(two_body)):
                    hijkl = two_body[i, j, k, l]*1/2
                    operators.append((two_body_ijk.compose(annihilation_operators[l]))*hijkl)

    qubit_hamiltonian = SparsePauliOp.sum(operators)
   
    return qubit_hamiltonian.simplify()

def minimize_expectation_value(observable: SparsePauliOp, ansatz: QuantumCircuit, starting_params: list, backend: Backend, minimizer: Callable, execute_opts: dict = {}) -> OptimizeResult:
    """
    Uses the minimizer to search for the minimal expection value of the observable for the
    state that the ansatz produces given some parameters.

    Args:
    observable (SparsePauliOp): The observable which the expectation value will be
    minimized.
    ansatz (QuantumCircuit): A paramtrized quantum circuit used to produce quantum state.
    starting_params (list): The initial parameter of the circuit used to start the
    minimization.
    backend (Backend): A Qiskit backend on which the cirucit will be executed.
    minimizer (Callable): A callable function, based on scipy.optimize.minimize which only
    takes a function and starting params as inputs.
    execute_opts (dict, optional): Options to be passed to the Qsikit execute function.

    Returns:
    OptimizeResult: The result of the optimization
    """

    def cost_function(params : list):
        state_circuit = ansatz.bind_parameters(params)
        expectation_values = estimate_expectation_values(observable.paulis, state_circuit, backend, execute_opts)
        return np.real(expectation_values.dot(observable.coeffs))

    result = minimizer(cost_function, starting_params, method = 'COBYLA')
    return result



def exact_minimal_eigenvalue(observable: SparsePauliOp) -> float:
    """
    Computes the minimal eigenvalue of an observable.
    Args:
    observable (SparsePauliOp): The observable to diagonalize.

    Returns:
    float: The minimal eigenvalue of the observable.
    """


    return min(np.linalg.eigvals(observable.to_matrix()))