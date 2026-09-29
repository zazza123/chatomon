# Deployment

## Overview

Guard models such as ScopeGuard and ClaimExtractor run as separate services, usually on a GPU. This section explains how chatomon reaches them when they sit behind private, authenticated endpoints.

## Table of Contents

| Document | Description |
|----------|-------------|
| [ScopeGuard on Cloud Run](cloud_run.md) | Calling a private Cloud Run service with refreshed identity tokens, cold starts and timeouts |

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `ScopeGuardOrbital`, `ClaimExtractorOrbital` | `src/chatomon/orbitals/principled.py` | Orbitals whose `headers` provider refreshes credentials before every request |
| Integration tests | `tests/integration/` | Opt-in tests against real deployments, driven by environment variables |

## Related Documentation

- [Principled Intelligence Orbitals](../orbitals/principled.md) — ScopeGuard, ClaimExtractor and their `headers` provider
- [Home](../index.md) — documentation overview

---
**Last Updated:** September 2026
**Status:** Implemented
