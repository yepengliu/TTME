"""Candidate evaluation: sandboxed execution, out-of-process scoring, local and remote evaluators.

The sandbox is best effort: candidates run as the calling user without a container, so real isolation
needs a container or a separate uid.
"""
