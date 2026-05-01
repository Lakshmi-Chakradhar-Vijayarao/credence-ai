---
name: Feature request
about: Suggest something new
labels: enhancement
---

**What problem does this solve?**
Describe the epistemic failure mode or workflow gap you're hitting.

**Proposed solution**
What would the new behavior look like?

**Which layer would this touch?**
- [ ] Faithfulness probe (new uncertainty markers / new domains)
- [ ] Consistency Enforcer (new synonym clusters)
- [ ] Generation-Time Scanner (new literal types)
- [ ] Rust gate (new tool matchers)
- [ ] Registry (new constraint types / decay rates)
- [ ] MCP server (new tool)
- [ ] Cross-session memory
- [ ] New layer entirely

**Would this require API calls?**
- [ ] No — purely deterministic / offline
- [ ] Yes — describe which model and approximate cost per call

**Alternatives considered**
Any other approaches you thought about.
