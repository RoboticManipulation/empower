#!/usr/bin/env python

import rospy
from visualization_msgs.msg import MarkerArray, Marker
import copy
import json
import os


class PersistentObjects:
    def __init__(self):
        # --- Subscribers ---
        rospy.Subscriber("/pcl_centroids", MarkerArray, self.cb_centroids)
        rospy.Subscriber("/pcl_names", MarkerArray, self.cb_names)

        # --- Publishers ---
        self.pub_centroids = rospy.Publisher(
            "/pcl_centroids_persistent", MarkerArray, queue_size=10, latch=True
        )
        self.pub_names = rospy.Publisher(
            "/pcl_names_persistent", MarkerArray, queue_size=10, latch=True
        )
        self.pub_graph = rospy.Publisher(
            "/pcl_graph", MarkerArray, queue_size=10, latch=True
        )

        # --- Internal storage ---
        self.objects = []   # list of centroid markers
        self.names = []     # list of text markers
        self.obj_map = {}   # map: centroid.id -> persistent name

        # --- Counter for unique IDs ---
        self.id = 1

        # --- JSON output path ---
        self.json_path =  "scenegraph.json"
        # Clean old persistent state at startup
        self.clear_persistent()

        rospy.loginfo("Persistent object daemon started.")

    def clear_persistent(self):
        """Clear old persistent objects, markers, and JSON on startup."""
        empty = MarkerArray()

        # Publish empty arrays to clear latched topics
        self.pub_centroids.publish(empty)
        self.pub_names.publish(empty)
        self.pub_graph.publish(empty)

        # Reset internal storage
        self.objects = []
        self.names = []
        self.obj_map = {}
        self.id = 1

        # Clear JSON file
        sg = {"nodes": [], "edges": []}
        with open(self.json_path, "w") as f:
            json.dump(sg, f, indent=2)

        rospy.loginfo("Cleared old persistent objects and reset scene graph.")

    # -------------------- CALLBACKS --------------------
    def cb_centroids(self, msg):
        updated = False
        for marker in msg.markers:
            # TODO: implement duplicate check (distance-based)
            duplicate = False
            if not duplicate:
                self.objects.append(copy.deepcopy(marker))
                rospy.loginfo(f"Added centroid marker with raw ID {marker.id}")
                updated = True

        if updated:
            self.publish_persistent()

    def cb_names(self, msg):
        updated = False
        for marker in msg.markers:
            if marker.id not in self.obj_map:
                base_label = marker.text if marker.text else "OBJECT"
                persistent_name = f"{base_label.upper()}_{self.id}"
                self.id += 1

                name_marker = copy.deepcopy(marker)
                name_marker.text = persistent_name

                self.names.append(name_marker)
                self.obj_map[marker.id] = persistent_name
                rospy.loginfo(f"Assigned new persistent name: {persistent_name}")
                updated = True

        if updated:
            self.publish_persistent()

    # -------------------- PUBLISH --------------------
    def publish_persistent(self):
        centroid_array = MarkerArray()
        name_array = MarkerArray()
        graph_array = MarkerArray()

        centroid_array.markers.extend(self.objects)
        name_array.markers.extend(self.names)

        # --- Identifica i ROOT (tavoli) ---
        roots = [obj for obj in self.names if "table" in obj.text.lower()]

        # --- Disegna i ROOT (sfere rosse + label) ---
        root_positions = {}
        for ridx, root in enumerate(roots):
            # posizione spostata in alto di +1m
            rp = copy.deepcopy(root.pose)
            rp.position.z += 1.0

            # --- Nodo ROOT (sfera) ---
            root_marker = Marker()
            root_marker.header.frame_id = "map"
            root_marker.header.stamp = rospy.Time.now()
            root_marker.ns = "scene_graph"
            root_marker.id = 1000 + ridx
            root_marker.type = Marker.SPHERE
            root_marker.action = Marker.ADD
            root_marker.pose = rp
            root_marker.scale.x = 0.15
            root_marker.scale.y = 0.15
            root_marker.scale.z = 0.15
            root_marker.color.r = 1.0
            root_marker.color.g = 0.0
            root_marker.color.b = 0.0
            root_marker.color.a = 1.0
            graph_array.markers.append(root_marker)

            # --- Nome del ROOT ---
            text_marker = copy.deepcopy(root_marker)
            text_marker.id = 2000 + ridx
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.text = root.text
            text_marker.scale.z = 0.2
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            graph_array.markers.append(text_marker)

            # salva posizione root spostata
            root_positions[root.text] = rp.position

        # --- Collega ogni oggetto al tavolo più vicino ---
        for idx, obj in enumerate(self.names):
            if any(obj.text == r.text for r in roots):
                continue  # è un root, skip

            # trova root più vicino
            dists = []
            for r in roots:
                rp = r.pose.position
                op = obj.pose.position
                d = (rp.x - op.x) ** 2 + (rp.y - op.y) ** 2 + (rp.z - op.z) ** 2
                dists.append((d, r))
            nearest_root = min(dists, key=lambda x: x[0])[1]
            root_pos = root_positions[nearest_root.text]

            # --- Edge (linea verde) ---
            edge_marker = Marker()
            edge_marker.header.frame_id = "map"
            edge_marker.header.stamp = rospy.Time.now()
            edge_marker.ns = "scene_graph_edges"
            edge_marker.id = 3000 + idx
            edge_marker.type = Marker.LINE_LIST
            edge_marker.action = Marker.ADD
            edge_marker.scale.x = 0.02
            edge_marker.color.r = 0.0
            edge_marker.color.g = 1.0
            edge_marker.color.b = 0.0
            edge_marker.color.a = 0.5
            edge_marker.points.append(root_pos)
            edge_marker.points.append(obj.pose.position)
            graph_array.markers.append(edge_marker)
        

        # --- Nodo ROOT globale (LAB) ---
        if roots:
            xs = [r.pose.position.x for r in roots]
            ys = [r.pose.position.y for r in roots]
            zs = [r.pose.position.z + 1 for r in roots]

            # centroide tra i tavoli
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            cz = max(zs) + 1.0   # 1m sopra il tavolo più alto

            lab_marker = Marker()
            lab_marker.header.frame_id = "map"
            lab_marker.header.stamp = rospy.Time.now()
            lab_marker.ns = "scene_graph"
            lab_marker.id = 9000
            lab_marker.type = Marker.SPHERE
            lab_marker.action = Marker.ADD
            lab_marker.pose.position.x = cx
            lab_marker.pose.position.y = cy
            lab_marker.pose.position.z = cz
            lab_marker.pose.orientation.w = 1.0
            lab_marker.scale.x = 0.2
            lab_marker.scale.y = 0.2
            lab_marker.scale.z = 0.2
            lab_marker.color.r = 0.0
            lab_marker.color.g = 0.0
            lab_marker.color.b = 1.0
            lab_marker.color.a = 1.0
            graph_array.markers.append(lab_marker)

            # --- Label del ROOT globale ---
            lab_text = copy.deepcopy(lab_marker)
            lab_text.id = 9001
            lab_text.type = Marker.TEXT_VIEW_FACING
            lab_text.text = "LAB"
            lab_text.scale.z = 0.3
            lab_text.color.r = 1.0
            lab_text.color.g = 1.0
            lab_text.color.b = 1.0
            graph_array.markers.append(lab_text)

            # --- Edges LAB → tavoli ---
            for tidx, root in enumerate(roots):
                edge = Marker()
                edge.header.frame_id = "map"
                edge.header.stamp = rospy.Time.now()
                edge.ns = "scene_graph_edges"
                edge.id = 9100 + tidx
                edge.type = Marker.LINE_LIST
                edge.action = Marker.ADD
                edge.scale.x = 0.03
                edge.color.r = 1.0
                edge.color.g = 1.0
                edge.color.b = 0.0
                edge.color.a = 0.8
                edge.points.append(lab_marker.pose.position)
                edge.points.append(root_positions[root.text])
                graph_array.markers.append(edge)



        
        # --- Publish all ---
        self.pub_centroids.publish(centroid_array)
        self.pub_names.publish(name_array)
        self.pub_graph.publish(graph_array)

        # --- Export JSON ---
        self.export_json(roots)

        rospy.loginfo(
            f"Published {len(roots)} ROOTs (tables), {len(self.names)} named objects."
        )


    # -------------------- EXPORT JSON --------------------
    def export_json(self, roots):
        sg = {"nodes": [], "edges": []}

        # ROOTs (tables)
        for root in roots:
            p = root.pose.position
            sg["nodes"].append(
                {
                    "id": root.text,
                    "category": "ROOT",
                    "position": [p.x, p.y, p.z],
                }
            )

        # Oggetti
        for obj in self.names:
            if any(obj.text == r.text for r in roots):
                continue  # è root

            p = obj.pose.position
            sg["nodes"].append(
                {"id": obj.text, "category": "OBJECT", "position": [p.x, p.y, p.z]}
            )

            # trova root più vicino
            dists = []
            for r in roots:
                rp = r.pose.position
                d = (rp.x - p.x) ** 2 + (rp.y - p.y) ** 2 + (rp.z - p.z) ** 2
                dists.append((d, r))
            nearest_root = min(dists, key=lambda x: x[0])[1]

            sg["edges"].append(
                {"source": nearest_root.text, "target": obj.text, "relation": "parent"}
            )

        with open(self.json_path, "w") as f:
            json.dump(sg, f, indent=2)

        rospy.loginfo(
            f"Exported scene graph JSON with {len(roots)} roots and {len(self.names)} objects."
        )

# -------------------- MAIN --------------------
if __name__ == "__main__":
    rospy.init_node("persistent_objects_daemon")
    node = PersistentObjects()
    rospy.spin()
