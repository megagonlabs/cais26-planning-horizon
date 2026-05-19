"""Freebase KB interaction utilities for KBQA tasks.

Submodules are imported lazily (at call site) to avoid requiring the
``pyodbc``/``libodbc.so.2`` system library when the Atomic KBQA track
is not in use (e.g. KoPL KBQA or Multi-objective HotpotQA experiments).
"""
