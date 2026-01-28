import json

import networkx as nx
import numpy as np
import open3d as o3d
from graphviz import Digraph
from plyfile import PlyData


# scenegraph3d class, with methods to load, dump and render graph
class SceneGraph3D:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.scan_id = None
        self.objects3d = None
        self.pointcloud = None

    def __getitem__(self, index):
        return self.graph.nodes[index]

    def nodes(self, data=False):
        return self.graph.nodes(data=data)

    def edges(self, data=False):
        return self.graph.edges(data=data)

    def search_by_name(self, name, all=True):
        # return all the nodes object that posses label equal to name
        results = []
        for id, node in self.nodes(data=True):
            if node["label"] == name:
                results.append(id)
                if not all:
                    break
        return results

    def get_neighbors_and_relation(self, node_id):
        neighbors = self.get_neighbors(node_id)
        e1 = node_id
        edges = self.graph.edges(data=True)
        neighbors_and_rels = []

        for neighbor in neighbors:
            e2 = neighbor
            for edge in edges:
                e_r1 = [edge[0]][0]
                e_r2 = [edge[1]][0]
                if e_r1 == e1 and e_r2 == neighbor:
                    neighbors_and_rels.append((neighbor, edge[2]))

        return neighbors_and_rels

    def get_neighbors(self, node_id):
        return list(self.graph.neighbors(node_id))

    def add_node(self, id, attribubtes_dict):
        self.graph.add_node(int(id), **attribubtes_dict)

    def read_3d_scene(self, file3d, pointcloud3dfile, mesh3dfile):
        # add the 3d informations about the scene given the semseg.v2.json file
        self.pointcloud = PlyData.read(open(pointcloud3dfile))
        self.mesh = o3d.io.read_triangle_mesh(mesh3dfile)

        data = json.load(open(file3d))
        # they must have the same scan_id
        assert data["scan_id"] == self.scan_id
        self.objects3d = {}
        for obj in data["segGroups"]:
            # keys: ['objectId', 'id', 'partId', 'index', 'dominantNormal', 'obb', 'segments', 'label']
            self.objects3d[obj["id"]] = obj

    def get_object_pointcloud(self, node_id, return_mask=False):
        # return the pointcloud of the object in 3d
        if self.pointcloud is None:
            return None
        vertex = self.pointcloud["vertex"]
        # get the points of the object
        xyz = np.vstack((vertex["x"], vertex["y"], vertex["z"])).T
        # get the colors of the object
        colors = np.vstack((vertex["red"], vertex["green"], vertex["blue"])).T
        object_ids = np.asarray(vertex["objectId"])
        obj_mask = object_ids == node_id
        # create pointcloud
        obj_colors = colors[obj_mask]
        obj_points = xyz[obj_mask]
        # create a new point cloud
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
        obj_pcd.colors = o3d.utility.Vector3dVector(obj_colors / 255.0)
        if return_mask:
            return obj_pcd, obj_mask
        return obj_pcd

    # def get_object_mesh(self, node_id):
    #     # return the mesh of the object in 3d
    #     if self.mesh is None or self.objects3d is None:
    #         return None
    #     obj = self.objects3d[node_id]
    #     segments = obj["segments"]
    #     segments = list(range(1, 101))
    #     mesh = self.mesh.select_by_index(segments, cleanup=True)
    #     return mesh

    def get_node_centroid(self, node_id):
        # return the centroid of the object in 3d
        if self.objects3d is None:
            return None
        return self.objects3d[node_id]["obb"]["centroid"]

    def get_closest_node(self, position):
        # return the closest node to the given position
        closest_node = None
        min_distance = float("inf")
        for id, node in self.nodes(data=True):
            centroid = self.get_node_centroid(id)
            distance = sum([(a - b) ** 2 for a, b in zip(centroid, position)])
            if distance < min_distance:
                min_distance = distance
                closest_node = id
        return closest_node

    @staticmethod
    def render(scenegraph, file):
        graph = Digraph()
        for id, o in scenegraph.nodes(data=True):
            graph.node(str(id), o["label"])
        for n1, n2, d in scenegraph.edges(data=True):
            graph.edge(str(n1), str(n2), label=d["name"])
        graph.render(file, view=False)

    @staticmethod
    def from_dict(graph_dict):
        # read scenegraph from dict
        scenegraph = SceneGraph3D()
        # scan id from 3RScans scan
        scenegraph.scan_id = graph_dict["scan"]
        objects = graph_dict["objects"]
        relationships = graph_dict["relationships"]
        for o in objects:
            scenegraph.graph.add_node(int(o["id"]), **o)
        for n1, n2, id, name in relationships:
            scenegraph.graph.add_edge(n1, n2, id=id, name=name)
        return scenegraph

    @staticmethod
    def to_dict(scenegraph):
        # dump scenegraph to dict
        objects = [o for id, o in scenegraph.nodes(data=True)]
        relationships = [
            (n1, n2, d["id"], d["name"]) for n1, n2, d in scenegraph.edges(data=True)
        ]
        return {
            "scan": scenegraph.scan_id,
            "objects": objects,
            "relationships": relationships,
        }

    @staticmethod
    def from_json(json_file):
        # load scenegraph from json file
        scenegraph_dict = json.load(open(json_file))
        return SceneGraph3D.from_dict(scenegraph_dict)

    @staticmethod
    def to_json(scenegraph, json_file):
        # dump scenegraph to json file
        scenegraph_dict = SceneGraph3D.to_dict(scenegraph)
        return json.dump(scenegraph_dict, open(json_file, "w"), indent=4)


def main():
    print("Test of the methods.")


if __name__ == "__main__":
    main()
