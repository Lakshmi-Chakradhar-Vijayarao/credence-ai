---
name: Bug report
about: Something isn't working as expected
labels: bug
---

**Which layer?**
- [ ] Faithfulness probe (CP1)
- [ ] Truth Buffer / Consistency Enforcer (CP2)
- [ ] Generation-Time Scanner (CP3)
- [ ] Rust gate (CP4)
- [ ] Cross-session memory (CP5)
- [ ] Registry / decay
- [ ] MCP server
- [ ] Other

**What happened?**
Describe the behavior you saw.

**What did you expect?**
What should have happened instead.

**Minimal reproduction**
```python
# Paste the smallest code that triggers the bug
```

**The uncertain constraint text** (if applicable)
```
# Paste the exact text that should have triggered / not triggered the probe
```

**Environment**
- OS:
- Python version:
- `pip show credence-guard` output:
- Rust gate built? (yes/no):

**Offline or API?**
- [ ] Offline (no API call involved)
- [ ] API call involved (model: ___ )
