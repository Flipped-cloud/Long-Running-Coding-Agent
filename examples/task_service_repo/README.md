# Task Service Repo

A small multi-file Python repository for v0.2 planning experiments.

## CLI Usage

```bash
# Add a task
python -m task_service.cli add --id 1 --title "My Task" --db tasks.json

# List all tasks
python -m task_service.cli list --db tasks.json

# Get a specific task by ID
python -m task_service.cli get --id 1 --db tasks.json

# Complete a task
python -m task_service.cli complete --id 1 --db tasks.json
```

The `get` command prints the task in the format `id: title completed=... attempts=...`
and exits with code 0. If the task ID is not found, it prints `task not found: <id>`
and exits with code 1.

## Run tests

```bash
python -m pytest -q
```
