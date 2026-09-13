"""Adapters that expose externally developed optimisers to the audit.

Each adapter imports its host library lazily, so :mod:`invaudit` itself never
depends on it.
"""
