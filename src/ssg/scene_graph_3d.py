import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import open3d as o3d
import torch
from torch import Tensor
from torch_geometric.data import Data

from graphviz import Digraph
from plyfile import PlyData
from consts import MESH_FILE, PCD_FILE, RSCAN_DIR, SEMSEG_FILE
from utils import invert_dict


CONVERTED_RELATIONSHIPS = {
    # none
    # supported by
    "left": "to the right of",
    "right": "to the left of",
    "front": "in front of",
    # behind": "behind",
    # close by": "close by",
    # inside": "inside",
    # bigger than": "bigger than",
    # smaller than": "smaller than",
    # higher than": "higher than",
    # lower than
    "same symmetry as": "the same symmetry as",
    "same as": "the same as",
    # attached to
    # standing on
    # lying on
    # hanging on
    # connected to
    # leaning against
    # part of
    # belonging to
    "build in": "built into",
    # standing in
    "cover": "covering",
    # lying in
    # hanging in
    "same color": "the same color as",
    "same material": "the same material as",
    "same texture": "the same texture as",
    "same shape": "the same shape as",
    "same state": "the same state as",
    "same object type": "the same object type as",
    # messier than
    # cleaner than
    # fuller than
    "more closed": "more closed than",
    "more open": "more open than",
    # brighter than
    # darker than
    # more comfortable than
}

INVERSE_CONVERTED_RELATIONSHIPS = invert_dict(CONVERTED_RELATIONSHIPS)


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

    def load_3d_data(self):
        file_3d_path: Path = RSCAN_DIR / self.scan_id / SEMSEG_FILE
        file_pointcloud: Path = RSCAN_DIR / self.scan_id / PCD_FILE
        file_mesh: Path = RSCAN_DIR / self.scan_id / MESH_FILE
        self.read_3d_scene(
            file_3d_path.as_posix(), file_pointcloud.as_posix(), file_mesh.as_posix()
        )

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
        converted = graph_dict.get("converted", False)
        for o in objects:
            scenegraph.graph.add_node(int(o["id"]), **o)
        for n1, n2, id, name in relationships:
            if converted:
                scenegraph.graph.add_edge(
                    n1, n2, id=id, name=INVERSE_CONVERTED_RELATIONSHIPS.get(name, name)
                )
            else:
                scenegraph.graph.add_edge(n1, n2, id=id, name=name)
        return scenegraph

    @staticmethod
    def to_dict(scenegraph, convert=False):
        # dump scenegraph to dict
        objects = [o for _, o in scenegraph.nodes(data=True)]
        relationships = [
            (
                n1,
                n2,
                d["id"],
                (
                    CONVERTED_RELATIONSHIPS.get(d["name"], d["name"])
                    if convert
                    else d["name"]
                ),
            )
            for n1, n2, d in scenegraph.edges(data=True)
        ]
        result = {
            "scan": scenegraph.scan_id,
            "objects": objects,
            "relationships": relationships,
        }
        if convert:
            result["converted"] = True
        return result

    @staticmethod
    def from_json(json_file):
        # load scenegraph from json file
        with open(json_file) as file:
            scenegraph_dict = json.load(file)
            return SceneGraph3D.from_dict(scenegraph_dict)
        return None

    @staticmethod
    def to_json(scenegraph, json_file, convert=False):
        # dump scenegraph to json file
        scenegraph_dict = SceneGraph3D.to_dict(scenegraph, convert=convert)
        return json.dump(scenegraph_dict, open(json_file, "w"), indent=4)

    @staticmethod
    def to_query_data(
        scenegraph,
        node_encoder: Callable[[Dict[str, Any]], Tensor],
        edge_encoder: Callable[[List[Dict[str, Any]]], Tensor],
        y_correct_nodes: Optional[List[str]] = None,
        ret_node_maps: bool = False,
    ):
        # convert scenegraph to pyg graph

        # produce a numerical index list for the nodes (preferrably matching their existing ids - 1)
        # node_ids maps from new 0-based node ids to old node ids
        node_map = scenegraph.graph.nodes
        node_map = list(map(str, node_map))

        inv_node_map = {str(node_map[i]): i for i in range(len(node_map))}
        assert len(node_map) == len(inv_node_map), "Node ids must be unique"

        x = torch.stack(
            [
                node_encoder(scenegraph.nodes(data=True)[int(node_map[node_id])])
                for node_id in range(len(node_map))
            ]
        )

        edge_index_map: Dict[Tuple[int, int], int] = {}
        edge_sources: List[int] = []
        edge_dests: List[int] = []
        edge_attr_args: List[List[Dict[str, Any]]] = []

        for n1, n2, d in scenegraph.edges(data=True):
            edge = (inv_node_map[str(n1)], inv_node_map[str(n2)])
            if edge in edge_index_map:
                edge_i = edge_index_map[edge]
                edge_attr_args[edge_i].append(d)
            else:
                edge_i = len(edge_attr_args)
                edge_index_map[edge] = edge_i
                edge_sources.append(edge[0])
                edge_dests.append(edge[1])
                edge_attr_args.append([d])

        edge_index = torch.tensor([edge_sources, edge_dests], dtype=torch.long)
        edge_attr = torch.stack([edge_encoder(args) for args in edge_attr_args])

        pos = []
        for node_id in node_map:
            node_pos = scenegraph.get_node_centroid(int(node_id))
            if node_pos is None:
                pos = None
                break
            pos.append(node_pos)

        if pos is not None:
            pos = torch.tensor(pos)

        y = None

        if y_correct_nodes is not None:
            y = torch.tensor([1 if node in y_correct_nodes else 0 for node in node_map])

        result = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)

        if ret_node_maps:
            result = (result, node_map, inv_node_map)
        return result


def main():
    print("Test of the methods.")


if __name__ == "__main__":
    main()
