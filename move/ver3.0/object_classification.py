import yaml
import os
import math

class SimpleObjectClassifier:
    def __init__(self, config_path='object_database.yaml'):
        self.objects = []
        self.margins = {'radius_margin_percent': 10, 'height_margin_percent': 10}
        self.load_database(config_path)

    def load_database(self, config_path):
        if not os.path.exists(config_path):
            print(f"[Classifier] Warning: {config_path} not found.")
            return

        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        self.objects = data.get('objects', [])
        self.margins = data.get('tolerance_settings', self.margins)
        print(f"[Classifier] Loaded {len(self.objects)} objects from database.")

    def _knn_distance(self, est_radius_cm, est_height_cm, obj):
        # 상대오차 기반 거리
        r = max(float(obj.get('radius', 0.0)), 1e-9)
        h = max(float(obj.get('height', 0.0)), 1e-9)

        rel_r = (est_radius_cm - r) / r
        rel_h = (est_height_cm - h) / h
        return math.sqrt(rel_r * rel_r + rel_h * rel_h)

    def identify(self, est_material, est_radius_cm, est_height_cm, k=1):
        """
        1) material 일치 후보만 뽑고
        2) 그 후보 내에서 (radius,height) kNN으로 가장 가까운 물체를 반환
        """
        print(f"\n[Classifier] Searching match for: {est_material}, r={est_radius_cm:.1f}cm, h={est_height_cm:.1f}cm")

        # 1) material 후보
        material_candidates = [
            obj for obj in self.objects
            if str(obj.get('material', '')).lower() == str(est_material).lower()
        ]
        if not material_candidates:
            return None, "Unknown Object (No material match in DB)"

        # 2) kNN
        neighbors = []
        for obj in material_candidates:
            dist = self._knn_distance(est_radius_cm, est_height_cm, obj)
            neighbors.append((dist, obj))

        neighbors.sort(key=lambda x: x[0])

        k = max(1, min(int(k), len(neighbors)))
        best_match = neighbors[0][1]
        best_dist = neighbors[0][0]

        # 디버그용: Top-k 표시
        topk_str = ", ".join(
            f"{n[1]['name']}(d={n[0]:.3f}, r={n[1]['radius']}, h={n[1]['height']})"
            for n in neighbors[:k]
        )

        result_str = f"Identified: {best_match['name']} (kNN d={best_dist:.3f} | Top-{k}: {topk_str})"
        return best_match, result_str