## What does this PR do?

<!-- One sentence. What changes and why. -->

## How to test

```bash
pytest tests/ -q
credence demo
```

## Checklist

- [ ] All tests pass (`pytest tests/ -q`)
- [ ] `credence demo` runs clean
- [ ] No new hard dependencies added to core package
- [ ] If changing decay rates in `registry.py`: S2 tests updated
- [ ] If adding uncertainty markers to `_UNCERTAINTY_MARKERS`: S22 tests updated
- [ ] If adding a new MCP tool: corresponding test suite added
