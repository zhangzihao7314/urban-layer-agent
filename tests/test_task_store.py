from src.web.task_store import (
    delete_task, duplicate_task, list_tasks, load_task, new_task, rename_task, save_task,
)


def test_task_lifecycle(tmp_path):
    task = new_task(tmp_path)
    task["messages"] = [{"role": "user", "content": "cool the city", "type": "text"}]
    task["decisions"] = {"7": "Dense trees"}
    save_task(tmp_path, task)
    assert load_task(tmp_path, task["task_id"])["decisions"]["7"] == "Dense trees"
    rename_task(tmp_path, task["task_id"], "Cooling")
    copy = duplicate_task(tmp_path, task["task_id"])
    assert copy["task_id"] != task["task_id"]
    assert len(list_tasks(tmp_path)) == 2
    delete_task(tmp_path, copy["task_id"])
    assert len(list_tasks(tmp_path)) == 1


def test_duplicate_copies_source_but_not_generated_outputs(tmp_path):
    task = new_task(tmp_path, "Original")
    source = tmp_path / "source.geojson"
    source.write_text('{"type":"FeatureCollection","features":[]}')
    task.update({
        "vector_path": str(source), "vector_name": source.name,
        "polygon_ids": [7], "goal": "cooling",
        "decisions": {"7": "Dense trees"},
        "outputs": {"updated_vector_path": "old/result.geojson"},
        "stage": "inputs_generated",
    })
    save_task(tmp_path, task)
    copy = duplicate_task(tmp_path, task["task_id"])
    assert copy["outputs"] == {}
    assert copy["stage"] == "ready_to_generate"
    assert copy["vector_path"] != str(source)
    assert (tmp_path / copy["task_id"] / "source" / source.name).exists()
