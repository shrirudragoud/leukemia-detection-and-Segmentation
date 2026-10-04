# Morning checklist (paper)

1. GPU results arrive as `results_for_paper/<gNN_name>/` (cv_summary.json, fold*_predictions.csv, fold*_train_log.jsonl).
2. `python scripts/finalize_paper.py --build --check`
   - fills the fine-tuning table (Table 10) from the result files,
   - replaces the three SAMPLE figures (tags FT-CURVES, FT-BARS, FT-CM) by the real plots (results_for_paper/figs/<TAG>.png),
   - lists everything unfinished (`[[GPU...]]`, `SAMPLE IMAGE`, `FILL IN`).
3. The three bracketed `[[GPU: ...]]` paragraphs (Results, Discussion) disappear once all eight result files exist; the text that replaces them
   is written from the numbers (ask the assistant: "write the fine-tuning results paragraphs from results_for_paper").
4. Fill `paper/meta.json` (authors, affiliation, correspondence, acknowledgments), then rebuild.
5. Tested end to end with synthetic results (paper/sample_results, made by scripts/make_sample_results.py): the pipeline runs without errors.
   The sample numbers are random and never appear in a build without the real files.
6. Verify the references listed in paper/REFERENCE_CHECKLIST.md.
