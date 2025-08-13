# 3D Quantum Topology Template

This template explores toy models inspired by 3D topological quantum error correction using small, CPU-friendly simulations. It integrates with AI Scientist and outputs the baseline artifacts required by the pipeline.

Key properties:
- Synthetic 3D lattice with i.i.d. “error/defect” noise (toy proxy for 3D code syndromes)
- Deterministic seeds, CPU-only, fast to run
- Baseline decoder/estimator: estimates logical error rate (LER) via 3D spanning-cluster probability
- Outputs final_info.json with {"accuracy": {"means": 1 - LER}} for pipeline compatibility

DISCLAIMER: This is a toy model for rapid research iteration; it is not a faithful simulator of any specific code and is not intended for hardware claims.

## Files
- experiment.py — Simulate 3D defects, estimate logical error rate vs. physical error p, write metrics and final_info.json
- plot.py — Plot LER vs p and a slice montage visualization of one configuration
- prompt.json — Guidance for LLM-driven iteration in 3D topological code-like settings
- seed_ideas.json — Starting ideas for exploration
- latex/template.tex — Base LaTeX template for write-up generation

## Quickstart

1) Baseline run
```
cd templates/quantum_topology_3d
python experiment.py --out_dir run_0
python plot.py --in_dir run_0
```

Artifacts:
- run_0/final_info.json (e.g., {"accuracy": {"means": 0.8}})
- run_0/metrics.json (LER vs p for one or more lattice sizes)
- run_0/curve.png, run_0/slices.png

2) Launch AI Scientist
Ensure your API keys are set as described in the project README, then:
```
cd ../../
python launch_scientist.py --experiment quantum_topology_3d --model "claude-3-5-sonnet-20240620" --num-ideas 1
```

## Configuration Tips
- Adjust lattice size via --L (default small), physical error grid via --p_min/--p_max/--p_steps, and trials per p via --trials.
- Use --tiny to run very fast smoke tests.
- All runs are deterministic given --seed.
