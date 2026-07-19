# Experience Learning Eval

This scaffold measures v0.4 knowledge behavior without relying on a live model.

It compares the five knowledge modes:

- `disabled`
- `raw_episode`
- `reflection`
- `verified_memory`
- `memory_skill`

Run:

```bash
python evals/experience_learning/runner.py --config evals/experience_learning/config.yaml
```

The report includes episode creation, memory retrieval hit rate, skill retrieval hit rate, negative transfer count, and verification success rate. The fixture repositories live under `examples/knowledge_transfer/`.
