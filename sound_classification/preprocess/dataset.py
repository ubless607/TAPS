
from pathlib import Path
from typing import List, Tuple


def get_all_audio_files(base_path: str = "Robot_impact_Data") -> List[Tuple[str, str, str]]:
    """
    모든 오디오 파일 경로를 수집.

    Returns
    -------
    file_list : list of (file_path, material, split)
        split ∈ {"train", "test"}
    """
    file_list: List[Tuple[str, str, str]] = []
    base = Path(base_path)

    # Horizontal_Pokes
    horizontal_path = base / "Horizontal_Pokes"
    if horizontal_path.exists():
        for material_dir in horizontal_path.iterdir():
            if not material_dir.is_dir():
                continue
            material = material_dir.name
            # Plastic 폴더의 경우 하위 폴더 처리 (Hard/Other/Soft 등)
            if material == "Plastic":
                for plastic_subdir in material_dir.iterdir():
                    if not plastic_subdir.is_dir():
                        continue
                    for split in ["train", "test"]:
                        split_path = plastic_subdir / split
                        if not split_path.exists():
                            continue
                        for obj_dir in split_path.iterdir():
                            if not obj_dir.is_dir():
                                continue
                            for audio_file in obj_dir.glob("*.ogg"):
                                file_list.append((str(audio_file), "Plastic", split))
            else:
                for split in ["train", "test"]:
                    split_path = material_dir / split
                    if not split_path.exists():
                        continue
                    for obj_dir in split_path.iterdir():
                        if not obj_dir.is_dir():
                            continue
                        for audio_file in obj_dir.glob("*.ogg"):
                            file_list.append((str(audio_file), material, split))

    # Vertical_Pokes / Known_Objects -> train
    vertical_known_path = base / "Vertical_Pokes" / "Known_Objects"
    if vertical_known_path.exists():
        for material_dir in vertical_known_path.iterdir():
            if not material_dir.is_dir():
                continue
            material = material_dir.name
            if material == "Plastic":
                for plastic_type in ["Hard", "Other", "Soft"]:
                    plastic_path = material_dir / plastic_type
                    if not plastic_path.exists():
                        continue
                    for obj_dir in plastic_path.iterdir():
                        if not obj_dir.is_dir():
                            continue
                        for audio_file in obj_dir.glob("*.ogx"):
                            file_list.append((str(audio_file), "Plastic", "train"))
            else:
                for obj_dir in material_dir.iterdir():
                    if not obj_dir.is_dir():
                        continue
                    for audio_file in obj_dir.glob("*.ogx"):
                        file_list.append((str(audio_file), material, "train"))

    # Vertical_Pokes / Unknown_Objects -> test
    vertical_unknown_path = base / "Vertical_Pokes" / "Unknown_Objects"
    if vertical_unknown_path.exists():
        for material_dir in vertical_unknown_path.iterdir():
            if not material_dir.is_dir():
                continue
            material = material_dir.name
            if material == "Plastic":
                for plastic_type in ["Hard", "Other", "Soft"]:
                    plastic_path = material_dir / plastic_type
                    if not plastic_path.exists():
                        continue
                    for obj_dir in plastic_path.iterdir():
                        if not obj_dir.is_dir():
                            continue
                        for audio_file in obj_dir.glob("*.ogx"):
                            file_list.append((str(audio_file), "Plastic", "test"))
            else:
                for obj_dir in material_dir.iterdir():
                    if not obj_dir.is_dir():
                        continue
                    for audio_file in obj_dir.glob("*.ogx"):
                        file_list.append((str(audio_file), material, "test"))

    return file_list

