# Neuromorphic BCI Template

This template provides a minimal, CPU-friendly baseline for neuromorphic Brain-Computer Interfaces (BCI) using synthetic spiking data and a lightweight linear decoder. It integrates with AI Scientist's pipeline and produces the baseline artifacts required by the orchestrator.

Key properties:
- Synthetic Poisson-like spikes with class-specific spatial and temporal structure
- Deterministic and fast (seconds on CPU)
- Minimal dependencies (uses numpy, torch, matplotlib)
- Outputs baseline `final_info.json` so AI Scientist can copy and evolve experiments

DISCLAIMER: This template uses synthetic data only and is not intended for clinical use or medical claims.

## Files
- experiment.py — Generate synthetic spikes, train a simple decoder, write metrics and final_info.json
- plot.py — Generate a raster plot and training metrics figure
- prompt.json — Guidance for LLM-driven iteration specific to neuromorphic BCI
- seed_ideas.json — A few starting ideas for ideation
- latex/template.tex — Base LaTeX template for write-up generation

## Quickstart

1) Baseline run
```
cd templates/neuromorphic_bci
# Faster tiny baseline
python experiment.py --out_dir run_0 --tiny
python plot.py --in_dir run_0

# Improved baseline with coarse temporal bins and 5 epochs
python experiment.py --out_dir run_bins --epochs 5 --agg binned --num_bins 4 --time_bins 200
python plot.py --in_dir run_bins
```

Artifacts:
- run_0 or run_bins/final_info.json (e.g., {"accuracy": {"means": 0.65}})
- run_0 or run_bins/metrics.json
- run_0 or run_bins/raster.png, metrics.png

Flags:
- --agg {sum, mean, flatten, binned}
- --num_bins N (used when --agg binned; recommended 3–5)
- --epochs default 5
- --max_channels_display for plot.py

2) Launch AI Scientist
Ensure your API keys are set as described in the project README, then:
```
cd ../../
python launch_scientist.py --experiment neuromorphic_bci --model "claude-3-5-sonnet-20240620" --num-ideas 1
```

## Configuration Tips
- Use `--tiny` for very fast runs.
- Adjust `--n_channels`, `--time_bins`, `--train_samples`, `--test_samples` to scale difficulty and runtime.
- Feature aggregation can be changed via `--agg {sum, mean, flatten}`.
