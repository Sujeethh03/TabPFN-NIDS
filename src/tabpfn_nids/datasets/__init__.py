"""Dataset loaders.

One module per dataset, each exposing a loader with the same signature so
that new datasets are added by writing a file rather than by editing a
conditional. ``REGISTRY`` maps a dataset name to its loader for CLI dispatch.
"""
