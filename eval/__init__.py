"""Evaluation harness: metrics, runners and plots.

Nothing in here imports a model. Metrics operate on stored logits, so a run can be
re-scored — recalibrated, re-bucketed, re-plotted — without touching a GPU.
"""
