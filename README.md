# AS02: LLM API for Legal Aid Clinic Precedent Research

Evaluation of Phi-4-mini-instruct for citation extraction and treatment classification on Australian Federal Court immigration opinions.

## Setup (Colab)

1. Open `AS02_llmbox_colab.ipynb` in Google Colab
2. **Before running:** Go to `Runtime → Change runtime type → GPU (T4)`
3. Run cells in order:
   - Cell 1: Confirm GPU is attached
   - Cell 2: Upload `llmbox_colab.zip` when prompted
   - Cell 3–4: Install dependencies and download Phi-4-mini-instruct weights (~7.7GB)
   - Cell 5–6: Set `LIMIT = 50` (or lower if short on time)
   - Cells 7+: Run evaluations (Part B baseline, Part C custom, Part D safety)

**Important:** Keep the Colab tab open while it's running. It disconnects after ~90 minutes of inactivity or if you close the tab.

## Outputs

Results are written per-record to `out/`:
- `eval1_responses.jsonl`, `eval1_summary.json` (Part B, high temperature)
- `eval3_responses.jsonl`, `eval3_summary.json` (Part B, low temperature)
- `structured_output_responses.jsonl`, `structured_output_summary.json` (Part C, schema)
- `tool_calling_responses.jsonl`, `tool_calling_summary.json` (Part C, tools)
- `partd_responses.jsonl`, `partd_adjudication.csv` (Part D, safety probes)

Last cell zips `out/` for download.

## Notes

- Model loads in **float16** (T4 has no bfloat16 support)
- Input capped at 6,000 tokens to fit in GPU memory
- ~36% of records hit OOM at this cap; OOM records return `[OOM]` instead of crashing
- One model instance; results are deterministic with seed 42

## Files Required

- `llmbox_colab.zip` — LLMBox src/, conf/, dev/eval sets, schemas, citation index
- `run_baseline_eval.py` — Part B baseline script (written by Cell 7)
- `run_custom_eval.py` — Part C custom script (written by Cell 8)

