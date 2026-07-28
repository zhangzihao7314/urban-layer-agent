"""Persistent, independent Urban Layer Agent tasks."""

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


def new_task(tasks_root: Path, title="New urban planning task") -> dict:
    task_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    task = {
        "task_id": task_id, "title": title,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": [], "vector_path": None, "vector_name": None,
        "id_column": None, "polygon_ids": [], "goal": "",
        "decisions": {}, "unchanged": [], "stage": "waiting_for_vector",
        "pending_proposal": None, "active_requirement": None,
        "outputs": {},
    }
    save_task(tasks_root, task)
    return task


def task_dir(tasks_root: Path, task_id: str) -> Path:
    return Path(tasks_root) / task_id


def save_task(tasks_root: Path, task: dict) -> Path:
    folder = task_dir(tasks_root, task["task_id"])
    folder.mkdir(parents=True, exist_ok=True)
    task["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = folder / "task.json"
    def json_default(value):
        if hasattr(value, "item"):
            return value.item()
        if hasattr(value, "value"):
            return value.value
        if isinstance(value, Path):
            return str(value)
        return str(value)
    temporary = folder / "task.json.tmp"
    temporary.write_text(
        json.dumps(task, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def load_task(tasks_root: Path, task_id: str) -> dict:
    return json.loads((task_dir(tasks_root, task_id) / "task.json").read_text(encoding="utf-8"))


def list_tasks(tasks_root: Path):
    tasks = []
    for path in Path(tasks_root).glob("*/task.json"):
        try:
            tasks.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(tasks, key=lambda item: item.get("updated_at", ""), reverse=True)


def rename_task(tasks_root: Path, task_id: str, title: str):
    task = load_task(tasks_root, task_id)
    task["title"] = title.strip() or task["title"]
    save_task(tasks_root, task)


def duplicate_task(tasks_root: Path, task_id: str):
    source = load_task(tasks_root, task_id)
    duplicate = new_task(tasks_root, f"{source['title']} (copy)")
    for key in ("messages", "vector_name", "id_column", "polygon_ids",
                "goal", "decisions", "unchanged"):
        duplicate[key] = source.get(key)
    source_vector = source.get("vector_path")
    if source_vector and Path(source_vector).exists():
        destination = task_dir(tasks_root, duplicate["task_id"]) / "source" / Path(source_vector).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_vector, destination)
        duplicate["vector_path"] = str(destination)
    covered = set(map(str, duplicate.get("decisions", {}))) | {
        str(item) for item in duplicate.get("unchanged", [])
    }
    polygon_ids = {str(item) for item in duplicate.get("polygon_ids", [])}
    if duplicate.get("vector_path") and duplicate.get("goal") and polygon_ids and covered == polygon_ids:
        duplicate["stage"] = "ready_to_generate"
    elif duplicate.get("vector_path") and duplicate.get("goal"):
        duplicate["stage"] = "waiting_for_polygon"
    elif duplicate.get("vector_path"):
        duplicate["stage"] = "waiting_for_goal"
    duplicate["pending_proposal"] = None
    duplicate["active_requirement"] = None
    duplicate["outputs"] = {}
    save_task(tasks_root, duplicate)
    return duplicate


def delete_task(tasks_root: Path, task_id: str):
    folder = task_dir(tasks_root, task_id)
    if folder.exists():
        shutil.rmtree(folder)


def title_from_goal(goal: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", goal)
    return " ".join(words[:6]).title() if words else "Urban planning task"
