#######################################################################
# Richardson extrapolation script for mesh study
# This script can be used to calculate the mesh-independent solution
# using Richardson Extrapolation. The included meshes are
# automatically sorted by cell number and only the three finest are
# used. Outputs are the mesh-independent solution and the error of the
# finest mesh
# Created by Kiiski
#
#  2025-11-06
# 
# Changelog:
#  Version 1.0:
#   - Initial release
#  Version 1.1:
#   - object-oriented
#  Version 1.2:
#   - Added implementation of GCI https://doi.org/10.1115/1.2960953
#
#######################################################################

import sys
import numpy as np

class MeshSimulation():
    """
    A Class including the mesh data. The convergence parameter can be
    added using the 'add_parameter' method.

    :param number_cells (int): The number of cells of the mesh
    :param volume (int, float): The volume of the domain

    :return: A MeshSimulation class
    """
    def __init__(self,
                 name,
                 number_cells,
                 volume):
        self.name = name
        self.number_cells = number_cells
        self.volume = volume
        self.cellsize = (self.volume / self.number_cells)**(1/3)
        self.parameters = {}
    
    def add_parameter(self, parameter, value):
        """
        A method to add convergence parameter values to a class. The
        parameters are stored in a dictionary.

        :param parameter (str): The parameter to be added
        :param value (int, float): The value of the parameter

        :return: nothing
        """
        self.parameters[parameter] = value

class RichardsonExtrapolation():
    """
    Class for performing the Richardson extrapolation.
    The class can be initiated with a list of at
    least THREE elements of the MeshSimulation class. It will perform
    the extrapolation on every element in the parameters dictionary of
    the meshes and save the mesh-independent solutions as well as the
    error for the finest provided mesh.
    :param list_of_meshes (list): A list of MeshSimulation classes
    :param num_iterations (int): The number of iterations used to
    calculate the mesh independent solution (default = 10)

    :return: A RichardsonExtrapolation class with mesh independent
    solutions and errors for the finest mesh
    """
    def __init__(self, list_of_meshes, num_iterations=100):
        self.meshes = list_of_meshes
        self.num_iterations = num_iterations
        self.meshes.sort(key=lambda mesh: mesh.number_cells, reverse = True)
        self.parameters = self.meshes[0].parameters

        if len(self.meshes) < 3:
            print('Please provide at least three meshes!\nAborting...')
            sys.exit(0)
        elif len(self.meshes) > 3:
            print('More than three meshes provided, only using the three'
                  ' finest')
            self.meshes = self.meshes[0:3]
        
        self.calculate_mesh_ratio()

        self.mesh_independent_solutions = {}
        self.gci_solutions = {}

    def richardson_glr(self):
        for parameter in self.parameters:
            p = self.calculate_p_and_q(parameter)
            mesh_values = np.array(
                [mesh.parameters[parameter] for mesh in self.meshes])
            mesh_independent = mesh_values[0] + (
                (mesh_values[0]-mesh_values[1]) / ((self.ratios[0]**p) - 1))
                    
            finest_error = ((mesh_values[0] - mesh_independent)
                            / mesh_independent)*1e2
            medium_error = ((mesh_values[1] - mesh_independent)
                            / mesh_independent)*1e2
            coarse_error = ((mesh_values[2] - mesh_independent)
                            / mesh_independent)*1e2
            self.mesh_independent_solutions[parameter] = (
                mesh_independent, finest_error, medium_error, coarse_error, p)

    def grid_convergence_index(self):
        for parameter in self.parameters:
            p = self.calculate_p_and_q(parameter)
            mesh_values = np.array(
                [mesh.parameters[parameter] for mesh in self.meshes])
            
            extrapolated_fine = (self.ratios[0]**p * mesh_values[0] -
                                 mesh_values[1])/(self.ratios[0]**p - 1)
            extrapolated_med = (self.ratios[0]**p * mesh_values[0] -
                                mesh_values[1])/(self.ratios[0]**p - 1)

            extrapolated_app_fine = np.abs((mesh_values[0]-mesh_values[1])
                                           /(mesh_values[0]))
            extrapolated_app_med = np.abs((mesh_values[1]-mesh_values[2])
                                          /(mesh_values[1]))

            extrapolated_rel_fine = np.abs((extrapolated_fine - mesh_values[0])
                                           /extrapolated_fine)
            extrapolated_rel_med = np.abs((extrapolated_med - mesh_values[1])
                                          /extrapolated_med)

            gci_fine = ((1.25 * extrapolated_app_fine)
                        / (self.ratios[0]**p - 1))
            gci_med = ((1.25 * extrapolated_app_med)
                       / (self.ratios[1]**p - 1))

            self.gci_solutions[parameter] = (
                extrapolated_fine, extrapolated_app_fine,
                extrapolated_rel_fine, gci_fine,
                extrapolated_med, extrapolated_app_med,
                extrapolated_rel_med, gci_med, p)

    def calculate_mesh_ratio(self):
        number_cells = np.array([mesh.cellsize for mesh in self.meshes])
        self.ratios = number_cells[1:] / number_cells[:-1]
    
    def calculate_mesh_error(self, parameter):
        mesh_values = np.array(
            [mesh.parameters[parameter] for mesh in self.meshes])
        mesh_errors = mesh_values[1:] - mesh_values[:-1]
        if mesh_errors[1]/mesh_errors[0] < 0:
            print(f'Negative mesh differences for {parameter}.\n'
                  'Probably oscillatory convervence!')
        elif (mesh_errors.min()) < 1e-4:
            print(f'Very low mesh differences for {parameter}.\n'
                  'Probably oscillatory convervence!')
        return mesh_errors
    
    def calculate_signs(self, errors):
        signs = np.sign(errors[1:]/errors[:-1])
        return signs

    def calculate_p_and_q(self, parameter):
        error = self.calculate_mesh_error(parameter)
        sign = self.calculate_signs(error)

        for ii in range(0,self.num_iterations):
            if ii == 0:
                q = 1/np.log(self.ratios[0]) * abs(np.log(abs(error[1]
                                                          / error[0])))
            else:
                q = (np.log(((self.ratios[0]**p_old) - sign[0])
                            / ((self.ratios[1]**p_old) - sign[0])))
            p = 1/np.log(self.ratios[0]) * abs(np.log(abs(error[1]
                                                          / error[0])) + q)
            p_old = p
        print(p)
        return p

    def print_gci(self):
        print('Grid Convergence Method results:')
        for key, (
                extrapolated_fine, extrapolated_app_fine,
                extrapolated_rel_fine, gci_fine,
                extrapolated_med, extrapolated_app_med,
                extrapolated_rel_med, gci_med, p
                ) in self.gci_solutions.items():

            print(f'{key}:\n'
                  f'Order of convergence: {p:.3f}\n'
                  'Extrapolated value:\n'
                  f'  Medium: {extrapolated_med}\n'
                  f'  Fine  : {extrapolated_fine}\n'
                  'Approximate relative error:\n'
                  f'  Medium: {extrapolated_app_med*1e2:.2f} %\n'
                  f'  Fine  : {extrapolated_app_fine*1e2:.2f} %\n'
                  'Extrapolated relative error:\n'
                  f'  Medium: {extrapolated_rel_med*1e2:.2f} %\n'
                  f'  Fine  : {extrapolated_rel_fine*1e2:.2f} %\n'
                  'Grid convergence index:\n'
                  f'  Medium: {gci_med*1e2:.2f} %\n'
                  f'  Fine  : {gci_fine*1e2:.2f} %\n')

    def print_mesh_independent(self):
        print('Mesh independent solutions are:')
        for key, value in self.mesh_independent_solutions.items():
            print(f'{key}:\n'
                  f'Mesh independent solution: {value[0]}\n'
                  f'Finest mesh solution: {self.meshes[0].parameters[key]}\n'
                  f'Error: {abs(value[1]):.3f} %\n'
                  f'Medium mesh solution: {self.meshes[1].parameters[key]}\n'
                  f'Error: {abs(value[2]):.3f} %\n'
                  f'Coarse mesh solution: {self.meshes[2].parameters[key]}\n'
                  f'Error: {abs(value[3]):.3f} %\n'
                  f'Order of convergence: {value[-1]:.3f}\n'
                  )
    
if __name__ == '__main__':
    
    mesh_data_file = None
    
    if mesh_data_file is not None:
        mesh_data = {}

        with open(mesh_data_file, 'r') as mesh_data_lines:
            for idx, line in enumerate(mesh_data_lines):
                elements = [e.strip().replace("'", "") for e in (
                    line.strip().split(','))]
                if idx == 0:
                    keys = elements
                    for key in keys:
                        mesh_data[key] = []
                else:
                    for key, value in zip(keys, elements):
                        try:value = int(value)
                        except ValueError:
                            try: value = float(value)
                            except ValueError:
                                value = value                    
                        mesh_data[key].append(value)
    else:
        mesh_data = {
            'Case': [1,2,3],
            'Elements': [18000, 8000, 4500],
            'Volume [m3]': [1,1,1],
            #'Dimensionless Reattachment Length': [6.063, 5.972, 5.863],
            'Axial Velocity at x/H = 8': [10.7880, 10.7250, 10.6050],
            #'Axial Velocity at x/H = 8, oscillatory': [6.0042, 5.9624, 6.0909]
        }

    meshes = []

    for idx, mesh in enumerate(mesh_data['Case']):
        meshes.append(
            MeshSimulation(
                mesh_data['Case'][idx],
                mesh_data['Elements'][idx],
                mesh_data['Volume [m3]'][idx])
        )
        
    used_meshes_idx = [0,1,2]
    used_meshes = []
    for idx in used_meshes_idx:
        used_meshes.append(meshes[idx])
    meshes = used_meshes
    mesh_cells = list(mesh.number_cells for mesh in meshes)
    print(mesh_cells)

    for idx, mesh in enumerate(meshes):
        for key in mesh_data:
            if key not in ['Case', 'Elements', 'Volume [m3]']:
                mesh.add_parameter(key, mesh_data[key][idx])

    extrap = RichardsonExtrapolation(meshes, 100)
    extrap.grid_convergence_index()
    extrap.richardson_glr()
    extrap.print_gci()
    extrap.print_mesh_independent()
