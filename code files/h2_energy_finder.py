from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from h2_energy_utils import build_quantum_state, find_energies, read_h2_file, afficher_graphique, find_min_energy
import numpy as np
import os
from typing import Callable
from qiskit.providers import Backend

def h2_min_energy(backend: Backend, minimizer : Callable, folder_path : str, execute_opts : dict = {}) -> tuple:
    """
    finds the h2 molecule minimal energy
    args: 
        backend: the backend on wich executing the circuits
        minimizer: the optimisation function to call to find the minimal energy
        folder_path: the path to the folder where the npz files containing the h2 molecule datasets are
        execute_opts: the options for the execution

    returns: 
        the minimal energy and the minimal distance for the h2 molecule
    """
    folder = os.listdir(folder_path)
    nb_elements = len(folder)
    distances = np.empty(nb_elements)
    energies = np.empty(nb_elements)
    min_energies = np.empty(nb_elements)
    quantum_state = build_quantum_state()

    #boucle d'itération de calcul de l'énergie minimale
    for i in range(nb_elements):
        distance, one_body, two_body, nuclear_repulsion_energy = read_h2_file(folder[i], folder_path)
        distances[i] = distance
        energy, min_energy = find_energies(quantum_state, backend, minimizer, one_body, two_body, nuclear_repulsion_energy, execute_opts)
        min_energies[i] = min_energy
        energies[i] = energy

        print("itération présente:", i+1, "/", nb_elements, "énergie:", np.round(energy, 4), "énergie minimale", np.round(min_energy, 4), "rayon:", distance)    

    #sort pour l'affichage
    order = np.argsort(distances)
    distances = distances[order]
    energies = energies[order]
    min_energies = min_energies[order]

    afficher_graphique(distances, energies, min_energies)
    min_energy, min_distance = find_min_energy(energies, distances)

    return min_energy, min_distance


folder_path = "h2_mo_integrals"
#éléments de calcul
backend = AerSimulator()
execute_opts = {"shots": 10000}

min_energy, min_distance = h2_min_energy(backend, minimize, folder_path, execute_opts)
print("energie minimale: ", min_energy, "rayon minimal: ", min_distance)

