#######################################################################
# Copyright (c) 2025 Yanneck Kiiski
# Licensed under the MIT License. See LICENSE file in the project root
# for full license information.
#
# This code is developed to unwrap the three-dimensionally curved
# surface of a turbine vane. It loads the mesh data of a vane or blade
# face and projects/parametrises it onto a new UV frame. Currently, this
# script works for blades or vanes that are left-hand twisted.
#
# It heavily depends on existing packages, with the main unwrapping
# being performed by the libigl LSCM solver. This script's main task is
# to properly preprocess the given mesh data and determine uv
# boundaries
#
# Version: 1.0
#
# ToDos:
# - reduce hard-coding for blade orientations
# - optimise corner detection
# - general cleanup and optimisation
# - comments
#
# Changelog:
# - converted to OOP
#
#######################################################################

import os
import json
import igl
import trimesh

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


### Class Definitions #################################################

class BladeSurface():
    def __init__(self, config_file):
        self.side = None
        if config_file is None:
            self.manual_flag = True
        else:
            self.manual_flag = False
        
    def get_mesh_from_pandas(self):
        vertices = self.node_data[[' X [ m ]', ' Y [ m ]',' Z [ m ]']].values
        faces = self.face_data[[0,1,2]].values

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.invert()

        self.trimesh = mesh.copy()
        self.V = mesh.vertices.astype(float)
        self.F = mesh.faces.astype(int)

    def load_mesh_config(self, case, blade, side):
        with open(config_file, 'r') as f:
                config = json.load(f)

        file = config[case]['File']
        self.side = side

        node_start, node_end = (
            config[case][blade][side]['Start Nodes']-1,
            config[case][blade][side]['End Nodes']-1)
        face_start, face_end = (
            config[case][blade][side]['Start Faces']-1,
            config[case][blade][side]['End Faces'])

        node_lines = node_end - node_start
        face_lines = face_end - face_start
        
        self.corners = np.array([
            config['Corners'][blade]['Leading Edge Top'],
            config['Corners'][blade][f'{side} Trailing Edge Top'],
            config['Corners'][blade][f'{side} Trailing Edge Bottom'],
            config['Corners'][blade]['Leading Edge Bottom']
            ])
        
        self.node_data = pd.read_csv(
            file, skip_blank_lines=False, index_col=False,
            header=node_start, nrows=node_lines)

        face_data = pd.read_csv(
            file, skip_blank_lines=False, index_col=False,
            skiprows=face_start, header=None, nrows=face_lines)
        self.face_data = face_data.astype(int)
        
        self.get_mesh_from_pandas()

    def load_mesh_manual(self, file,
                         nodes_start, nodes_nrows,
                         faces_start, faces_nrows,
                         corners):
        
        self.corners = corners
    
        self.node_data = pd.read_csv(
            file, skip_blank_lines=False, index_col=False,
            header=nodes_start, nrows=nodes_nrows)

        face_data = pd.read_csv(
            file, skip_blank_lines=False, index_col=False,
            skiprows=faces_start, header=None, nrows=faces_nrows)
        self.face_data = face_data.astype(int)
        
        self.get_mesh_from_pandas()

    def get_boundaries(self, watertight=None):
        if watertight is None:
            self.watertight_flag = self.trimesh.is_watertight
        else:
            self.watertight_flag = watertight
        
        unique_edges = self.trimesh.edges[
            trimesh.grouping.group_rows(self.trimesh.edges_sorted,
                                        require_count=1)
                                        ]
        
        if self.watertight_flag is True:
            print('Mesh is watertight!')
            self.ordered_boundary = self.order_edges(unique_edges)
        else:
            print('Mesh is not watertight, attempting to close holes!')
            boundary_loops = self.find_connected_components(unique_edges)
            boundary_loop_lengths = []
            for boundary_loop in boundary_loops:
                loop_edges = unique_edges[np.isin(unique_edges[:, 0],
                                                  boundary_loop)]
                loop_length = np.sum(np.linalg.norm(
                    self.V[loop_edges[:, 0]] - self.V[loop_edges[:, 1]],
                    axis=1))
                boundary_loop_lengths.append(loop_length)
            
            largest_loop_idx = np.argmax(boundary_loop_lengths)
            self.boundary_loops = boundary_loops
            self.main_boundary_idx = largest_loop_idx
            print(f'Found {len(boundary_loops)-1} holes!')
            self.number_vertices = len(self.V)
            new_verts = []
            patch_vertices = []
            patch_faces = []
            print(f'Patching holes...', end='')
            i, j = 0, 0
            for boundary_loop in boundary_loops:
                if i != largest_loop_idx:
                    j += 1
                    print(f'Hole {j} of {len(boundary_loops)-1}')
                    loop_edges = unique_edges[np.isin(unique_edges[:, 0],
                                                      boundary_loop)]
                    loop_center, loop_patch_faces = (
                        self.patch_circular_holes(loop_edges, self.V))
                    patch_vertices.append(loop_center)
                    patch_faces.append(loop_patch_faces)
                    new_verts.append(loop_center)
                    self.V = np.vstack([self.V, loop_center])
                else:
                    outer_loop_edges = unique_edges[np.isin(
                        unique_edges[:, 0], boundary_loop)]
                    self.ordered_boundary = self.order_edges(
                        outer_loop_edges)
                i += 1
            
            print('Done!')
            stacked_patch_faces =  np.vstack(patch_faces)
            self.hole_patches = stacked_patch_faces
            self.patch_verts = np.stack(new_verts)
            self.F = np.vstack([self.F, stacked_patch_faces])

            print('Detecting corners...')
            self.corner_ids = []
            self.corner_ids_abs = []
            boundary_verts = self.V[self.ordered_boundary]
            for corner in self.corners:
                distance_to_corner = np.linalg.norm(boundary_verts - corner, axis=1)
                closest_on_boundary = np.argmin(distance_to_corner)
                self.corner_ids.append(closest_on_boundary)
                self.corner_ids_abs.append(self.ordered_boundary[closest_on_boundary])
        self.unique_edges = unique_edges

    def cut_edges(self, rev=None):
        if self.side is None:
            if rev is None:
                rev = False
        else:
            if self.side == 'SS':
                rev = False
            elif self.side == 'PS':
                rev = True

        if rev == False:
            direction = [1,0,2,1,3,2,0,3]
        else:
            direction = [0,1,1,2,2,3,3,0]

        self.top = self.drop_duplicates(
            self.ordered_boundary[
                self.seg_positions(
                    self.corner_ids[direction[0]],
                    self.corner_ids[direction[1]],
                    len(self.ordered_boundary)
                    )])
        self.te = self.drop_duplicates(
            self.ordered_boundary[
                self.seg_positions(
                    self.corner_ids[direction[2]],
                    self.corner_ids[direction[3]],
                    len(self.ordered_boundary)
                    )])
        self.bottom = self.drop_duplicates(
            self.ordered_boundary[
                self.seg_positions(
                    self.corner_ids[direction[4]],
                    self.corner_ids[direction[5]],
                    len(self.ordered_boundary)
                    )])
        self.le = self.drop_duplicates(
            self.ordered_boundary[
                self.seg_positions(
                    self.corner_ids[direction[6]],
                    self.corner_ids[direction[7]],
                    len(self.ordered_boundary)
                    )])

    def patch_circular_holes(self, edges, vertices):
    
        hole_loop_nodes = self.order_edges(edges)
        if hole_loop_nodes[0] == hole_loop_nodes[-1]:
            hole_loop_nodes = hole_loop_nodes[:-1]

        edge_vertices = vertices[hole_loop_nodes]
        center_vertex = np.mean(edge_vertices, axis=0)
        center_index = len(vertices)
        new_faces = []
        for i in range(len(hole_loop_nodes)):
            v1 = hole_loop_nodes[i]
            v2 = hole_loop_nodes[(i + 1) % len(hole_loop_nodes)]
            new_faces.append([v1, v2, center_index])

        new_faces = np.array(new_faces, dtype=int)
        
        return center_vertex, new_faces

    def build_uv_frame(self, rev=None):
        if self.side is None:
            if rev is None:
                rev = False
        else:
            if self.side == 'SS':
                rev = False
            elif self.side == 'PS':
                rev = True

        if rev == False:
            boundary_loop = [self.bottom, self.te, self.top, self.le]
            u_order = [(0,1), (1,1), (1,0), (0,0)]
            v_order = [(0,0), (0,1), (1,1), (1,0)]
        else:
            boundary_loop = [self.le, self.top, self.te, self.bottom]
            u_order = [(0,0), (0,-1), (-1,-1), (-1,0)]
            v_order = [(0,1), (1,1), (1,0), (0,0)]

        uv_boundary_v = []
        uv_boundary_u = []

        for (boundary_segment,
             (u_min, u_max),
             (v_min, v_max)) in zip(boundary_loop, u_order, v_order):
            segment_verts = self.V[boundary_segment]
            distances = np.linalg.norm(np.diff(segment_verts, axis=0),
                                       axis=1)
    
            cumulative_distances = np.cumsum(np.insert(distances, 0, 0))
 
            total_length = cumulative_distances[-1]
            if total_length > 0:
                normalized_distances = cumulative_distances / total_length
            else:
                normalized_distances = np.zeros_like(cumulative_distances)

            u_array = u_min + (u_max - u_min) * normalized_distances
            v_array = v_min + (v_max - v_min) * normalized_distances
        
            uv_boundary_u.append(u_array)
            uv_boundary_v.append(v_array)

        uv_boundary_u_trim = [
            uv_boundary_u[0][1:-1],
            uv_boundary_u[1],
            uv_boundary_u[2][1:-1],
            uv_boundary_u[3]
        ]

        uv_boundary_v_trim = [
            uv_boundary_v[0][1:-1],
            uv_boundary_v[1],
            uv_boundary_v[2][1:-1],
            uv_boundary_v[3]
        ]

        boundary_trim = [
            boundary_loop[0][1:-1],
            boundary_loop[1],
            boundary_loop[2][1:-1],
            boundary_loop[3]
        ]
        u = np.concatenate(uv_boundary_u_trim)
        v = np.concatenate(uv_boundary_v_trim)

        self.segmented_boundary = np.concatenate(boundary_trim)
        self.uv_coords = np.vstack((u,v)).T

    def unwrap_lscm(self):
        print('Using libigl LSCM to parametrise surface')
        uv, success = igl.lscm(
            self.V, self.F,
            self.segmented_boundary,
            self.uv_coords)
        
        hole_uv_polygons = []
        if self.watertight_flag is False:
            for idx, component in enumerate(self.boundary_loops):
                if idx != self.main_boundary_idx:
                    hole_edges = self.unique_edges[np.isin(
                        self.unique_edges[:, 0], component)]
                    hole_loop_nodes = self.order_edges(hole_edges)
                    hole_uv_coords = uv[hole_loop_nodes].tolist()
                    hole_uv_polygons.append(hole_uv_coords)
                    
            face_mask = np.all(self.F < self.number_vertices, axis=1)
            self.F = self.F[face_mask]
            self.V = self.V[:self.number_vertices]
            self.uv = uv[:self.number_vertices]
            self.hole_poly = hole_uv_polygons

    def save_json(self, file_name):
        blade_unwrap ={
        'X' : list(self.node_data[' X [ m ]']),
        'Y' : list(self.node_data[' Y [ m ]']),
        'Z' : list(self.node_data[' Z [ m ]']),
        'U' : list(self.uv[:,0]),
        'V' : list(self.uv[:,1]),
        'Holes': self.hole_poly
        }
    
        if os.path.exists(file_name):
            print('File exists, appending!')
            with open(file_name, 'r+') as f:
                file_data = json.load(f)
                if case in file_data.keys():
                    if blade in file_data[case].keys():
                        file_data[case][blade][side] = blade_unwrap
                    else:
                        file_data[case][blade] = {
                            side: blade_unwrap
                        }
                else:
                    file_data[case] = {
                        blade: {
                            side: blade_unwrap
                        }
                    }
                    file_data
                f.seek(0)
                json.dump(file_data, f, indent=4)
        else:
            print('Creating new file')
            with open(file_name, 'w') as f:
                file_data = {
                    case:{
                        blade:blade_unwrap
                    }
                }
                json.dump(file_data, f, indent=4)

    def plot_trimesh(self):
        self.trimesh.show()

    def plot_boundaries(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        if self.watertight_flag is False:
            ax.add_collection3d(Poly3DCollection(
                self.V[self.hole_patches],
                facecolors='gold',
                alpha=0.5,
                label='Hole Faces'))
            ax.scatter(
                self.patch_verts[:,0],
                self.patch_verts[:,1],
                self.patch_verts[:,2],
                color='orange', s=25, marker='1',
                label='Hole Vertices'
            )
        
        top_verts = self.V[self.top]
        bottom_verts = self.V[self.bottom]
        le_verts = self.V[self.le]
        te_verts = self.V[self.te]
        corner_verts = self.V[self.corner_ids_abs]

        ax.scatter(top_verts[:,0],
                   top_verts[:,1],
                   top_verts[:,2],
                   color='lime', s=25, marker='2',
                   label='Top Vertices')
        ax.scatter(bottom_verts[:,0],
                   bottom_verts[:,1],
                   bottom_verts[:,2],
                   color='seagreen', s=25, marker='2',
                   label='Bottom Vertices')
        ax.scatter(le_verts[:,0],
                   le_verts[:,1],
                   le_verts[:,2],
                   color='forestgreen', s=25, marker='2',
                   label='Leading Edge Vertices')
        ax.scatter(te_verts[:,0],
                   te_verts[:,1],
                   te_verts[:,2],
                   color='springgreen', s=25, marker='2',
                   label='Trailing Edge Vertices')
        
        ax.scatter(corner_verts[:,0],
                   corner_verts[:,1],
                   corner_verts[:,2],
                   color='deepskyblue', s=25, marker='2',
                   label='Corner Vertices')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_uv_coords(self):
    
        segmented_boundary_xyz = self.V[self.segmented_boundary]
        u_colors = self.uv_coords[:, 0]
        v_colors = self.uv_coords[:, 1]

        fig = plt.figure()
        
        ax1 = fig.add_subplot(121, projection='3d')
        sc1 = ax1.scatter(segmented_boundary_xyz[:, 0], 
                        segmented_boundary_xyz[:, 1], 
                        segmented_boundary_xyz[:, 2], 
                        c=u_colors, cmap='viridis', s=10)

        ax1.set_title("U coordinate on boundary loop")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        plt.colorbar(sc1, ax=ax1, label="U")
        
        ax2 = fig.add_subplot(122, projection='3d')
        sc2 = ax2.scatter(segmented_boundary_xyz[:, 0], 
                        segmented_boundary_xyz[:, 1], 
                        segmented_boundary_xyz[:, 2], 
                        c=v_colors, cmap='viridis', s=10)
        ax2.set_title("V coordinate on boundary loop")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        plt.colorbar(sc2, ax=ax2, label="V")

        plt.tight_layout()
        plt.show()

    def plot_unwrapped_uv(self):
        uv_u = self.uv[:,0]
        uv_v = self.uv[:,1]

        plt.figure()
        plt.triplot(uv_u, uv_v, self.F, lw=0.5)
        plt.axis('equal')
        plt.title("Flattened Surface (UV Space)")
        plt.show()

    @staticmethod
    def order_edges(edges):
        ordered = [edges[0, 0], edges[0, 1]]
        edges = edges.tolist()
        edges.pop(0)
        while edges:
            last = ordered[-1]
            for i, (a, b) in enumerate(edges):
                if a == last:
                    ordered.append(b)
                    edges.pop(i)
                    break
                elif b == last:
                    ordered.append(a)
                    edges.pop(i)
                    break
            else:
                break
        return np.array(ordered)
    
    @staticmethod
    def find_connected_components(edges):
        adjacency = defaultdict(list)
        for a, b in edges:
            adjacency[a].append(b)
            adjacency[b].append(a)
        visited = set()
        connected_components = []

        for node in adjacency:
            if node not in visited:
                component = []
                queue = deque([node])
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        queue.extend(adjacency[current])
                connected_components.append(component)
        return connected_components
  
    @staticmethod
    def seg_positions(s, e, N):
        s = int(s); e = int(e)
        if s <= e:
            return np.arange(s, e + 1, dtype=int)  # Inclusive range
        return np.concatenate((np.arange(s, N, dtype=int), np.arange(0, e + 1, dtype=int)))
    
    @staticmethod
    def drop_duplicates(array):
        _, idx = np.unique(array, return_index=True)
        return array[np.sort(idx)]

### Stand alone #######################################################

if __name__ == '__main__':

    ### Parameters ####################################################

    ### config from file
    use_config_file = True
    config_file = 'blade_unwrap_config.json'

    # surface to unwrap
    case = 'Cooled'
    blade = 'Blade2'
    side = 'SS'

    ### manual config
    # file with mesh coordinates
    mesh_csv = 'example_mesh.csv'

    # ids for nodes and faces, from cfx: nodes with header, faces not
    start_read_nodes_idx = 0
    len_nodes_lines = 0

    start_read_faces_idx = 0
    len_faces_lines = 0

    # provide corner points of blade (does not need to be exact)
    leading_edge_bottom = [0,0,0]
    leading_edge_top = [0,0,0]
    trailing_edge_bottom = [0,0,0]
    trailing_edge_top = [0,0,0]
    reverse_corners = False

    ### additional settings
    # override watertightness for hole patching
    watertight_override = None

    save_flag = True
    save_file = 'test_uv.json'

    ### Plotting options ##############################################

    # these plotting options are mainly included for debugging and
    # verification of operations
    plot_loaded_mesh = False
    plot_boundaries = False
    plot_uv_frame = False
    plot_uv = False
    plot_uv_on_3d = False

    ###################################################################
    if use_config_file is False:
        corners = np.array([
            leading_edge_top,
            trailing_edge_top,
            trailing_edge_bottom,
            leading_edge_bottom
        ])

        blade_data = BladeSurface(None)
        blade_data.load_mesh_manual(
            file=mesh_csv,
            nodes_start=start_read_nodes_idx,
            nodes_nrows=len_nodes_lines,
            faces_start=start_read_faces_idx,
            faces_nrows=len_faces_lines,
            corners=corners
        )

    elif use_config_file is True:
        blade_data = BladeSurface(config_file=config_file)
        blade_data.load_mesh_config(
            case=case,
            blade=blade,
            side=side
        )
    
    if plot_loaded_mesh is True:
        blade_data.plot_trimesh()

    blade_data.get_boundaries(watertight_override)
    blade_data.cut_edges(reverse_corners)

    if plot_boundaries is True:
        blade_data.plot_boundaries()
    
    blade_data.build_uv_frame()

    if plot_uv_frame is True:
        blade_data.plot_uv_coords()
    
    blade_data.unwrap_lscm()

    if plot_uv is True:
        blade_data.plot_unwrapped_uv()

    if save_flag is True:
        blade_data.save_json(save_file)
