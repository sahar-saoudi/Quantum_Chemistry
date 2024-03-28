from Quantum_Hamiltonian import annihilation_operators_with_jordan_wigner, build_qubit_hamiltonian, minimize_expectation_value, exact_minimal_eigenvalue
from qiskit.providers import Backend
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
import os, re
from typing import List, Callable
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def read_h2_file(filename : str, folder_path : str) -> tuple:
    """
    reads the file specified in the arguments if it matches the specified condition 
    args:
        filename: name of the file to be read
        folder_path: path to the folder where the file is located
    returns: 
        information contained in the folder that is required for finding the energy of the h2 molecule
    """
    match = re.match(r"h2_mo_integrals_d_(\d{4})\.npz", filename)
    if (match):
        filepath = os.path.join(folder_path, filename)
        npzfile = np.load(filepath)
        return npzfile["distance"], npzfile["one_body"], npzfile["two_body"], npzfile["nuclear_repulsion_energy"]


def build_quantum_state() -> QuantumCircuit:
    """
    build the quantum state that represents the h2 molecule

    returns: 
        quantum circuit that represents the quantum state
    """
    qreg = QuantumRegister(4, "q")
    theta = Parameter("t")
    quantum_state = QuantumCircuit(qreg)
    quantum_state.ry(theta, qreg[1])
    quantum_state.x(qreg[0])
    quantum_state.cx(qreg[1], qreg[0])
    quantum_state.cx(qreg[0], qreg[2])
    quantum_state.cx(qreg[1], qreg[3])
    return quantum_state


def afficher_graphique(distances: NDArray, energies: NDArray, min_energies: NDArray):  
    """
    plots a graph of all the energies and the minimal energies for the rayon

    args:
        distances: array containing the rayons
        energies: array containing the energies finded
        min_energies: array containing the minimal energies
    """
    plt.plot(distances, energies, "r", label = "énergie minimale")
    plt.plot(distances, min_energies, "g", label = "énergie minimale attendue")
    plt.xlabel("rayon moléculaire")
    plt.ylabel("energie de la molécule")
    plt.title("energie de la molécule de H2 en fonction du rayon")
    plt.legend()
    plt.savefig("energie_graph")

def find_min_energy(energies: NDArray, distances: NDArray) -> tuple:
    """
    finds the minimal energy in an array of energies

    args:
        energies: array containing the energies to find the minimal value
        distances: array containing the distances for the energies

    returns: 
        the minimal energy in the array and the distance for that given energy
    """
    min_energy = np.min(energies)
    min_distance = distances[np.where(energies == min_energy)]
    return min_energy, min_distance

def find_energies(quantum_state : QuantumCircuit, backend : Backend, minimizer : Callable, one_body : List, two_body : List, nuclear_repulsion_energy : float, execute_opts : dict = {}) -> tuple:
    """
    finds the energy and the expected energy of h2 for a given distance

    args:
        quantum_state: quantum circuit representing the h2 molecule
        backend: the backend to execute the circuit on
        minimizer: the function to call to minimize the energy
        one_body: an array representing the one_body for the given distance
        two_body: an array representing the two body for the given distance
        nuclear_repulsion_energy: the nuclear_repulsion_energy for the given distance
        execute_opts: the options for the execution of the circuits

    returns:
        the obtained minimal energy and the expected minimal energy for a given distance
    """

    annihilation_operators = annihilation_operators_with_jordan_wigner(len(one_body))
    creation_operators = [a.adjoint() for a in annihilation_operators]

    qubit_hamiltonian = build_qubit_hamiltonian(one_body, two_body, annihilation_operators, creation_operators)
    obtained_min = minimize_expectation_value(qubit_hamiltonian, quantum_state, [0], backend, minimizer, execute_opts).fun
    expected_min = exact_minimal_eigenvalue(qubit_hamiltonian)

    energy = obtained_min + nuclear_repulsion_energy
    min_energy = np.real(expected_min + nuclear_repulsion_energy)

    return energy, min_energy



