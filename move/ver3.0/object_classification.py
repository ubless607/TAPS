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

    def identify(self, est_material, est_radius_cm, est_height_cm):
        """
        Identify object based on material and dimensions with tolerance.
        Returns: (Best Match Object Dict, match_details_string)
        """
        candidates = []
        
        print(f"\n[Classifier] Searching match for: {est_material}, r={est_radius_cm:.1f}cm, h={est_height_cm:.1f}cm")

        for obj in self.objects:
            # 1. Material Filter (Case insensitive)
            if obj['material'].lower() != est_material.lower():
                continue

            # 2. Dimension Check (with Percentage-based Margins)
            rad_diff = abs(obj['radius'] - est_radius_cm)
            h_diff = abs(obj['height'] - est_height_cm)
            
            # Calculate allowable margin as percentage of actual object dimension
            rad_margin = obj['radius'] * (self.margins['radius_margin_percent'] / 100.0)
            h_margin = obj['height'] * (self.margins['height_margin_percent'] / 100.0)

            if rad_diff <= rad_margin and h_diff <= h_margin:
                # Calculate a simple error score (lower is better)
                # We weight them equally here, but you can adjust weights.
                score = rad_diff + h_diff
                candidates.append((score, obj))

        if not candidates:
            return None, "Unknown Object (No match in DB)"

        # 3. Sort by lowest error score
        candidates.sort(key=lambda x: x[0])
        best_match = candidates[0][1]
        score = candidates[0][0]

        result_str = (f"Identified: {best_match['name']} "
                      f"(Err: {score:.2f} | DB: r{best_match['radius']}/h{best_match['height']})")
        
        return best_match, result_str