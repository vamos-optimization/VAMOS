"""Experimental NiceGUI interface for VAMOS (``vamos gui``).

The interface is a thin client of canonical artifacts and the public run API:

- ``events``, ``jobs``, ``worker``, ``monitor``, ``catalog``, and ``tables`` are
  UI-independent services (progress events, out-of-process runs, read-only run
  discovery, table/CSV adapters) and never import NiceGUI;
- ``app`` builds the NiceGUI pages and ``cli`` parses ``vamos gui`` arguments.

Install the optional dependencies with ``pip install "vamos-optimization[gui]"``.
"""
