# Security policy

## Supported versions

Security fixes are released for the latest version only.

| Version | Supported |
|---|---|
| latest | ✅ |
| older | ❌ |

## Reporting a vulnerability

**Do not report security vulnerabilities in public issues, discussions or pull
requests.**

Report them privately through either channel:

- **GitHub Security Advisories** (preferred): the
  [*Report a vulnerability*](https://github.com/zazza123/chatomon/security/advisories/new)
  button in the Security tab of this repository.
- **Email**: `andrea.zanini@andromedalabs.io`, with the subject
  `[chatomon][security] <short description>`.

Please include:

- what the issue is and what an attacker could achieve with it;
- steps to reproduce, or a proof of concept;
- the affected chatomon version, Python version and agent framework;
- a suggested mitigation, if you have one.

chatomon is a guardrail library: a way to **bypass an orbital** — a message that
reaches the model or the user although an orbital should have blocked it — is a
security issue and belongs here.

## What happens next

- An acknowledgement within **7 days**.
- A fix or a mitigation plan within **30 days**, depending on severity and
  complexity.
- A disclosure date agreed with you before any advisory or release notes are
  published.

Thank you for helping keep chatomon and its users safe.
