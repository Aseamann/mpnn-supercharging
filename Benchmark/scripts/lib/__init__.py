"""Shared benchmark library.

This is a package so that lib/io.py resolves as `lib.io` and does not collide
with the standard library's `io`, which the interpreter has already imported
before any script runs.
"""
