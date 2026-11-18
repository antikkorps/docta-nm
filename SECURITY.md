# Security policy

## Reporting a vulnerability

Please do **not** open a public issue for a security problem. Report it
privately instead, through GitHub:
**Security → Report a vulnerability** on
[this repository](https://github.com/antikkorps/docta-nm/security/advisories/new).

Say what you found, how to reproduce it, and what it could affect. You will get
an answer as soon as possible; a fix and a public advisory follow once the
problem is understood.

## Supported versions

Docta-nm is a research project at version `0.0.x`. Only the latest release and
the `dev` branch receive fixes.

## Scope worth reporting

- Anything that lets ground truth (`facts`, `fact_ids`, gold answers) reach a
  model through the search tool or the orchestrator.
- Injection through queries or loaded data (SQL, full-text query syntax).
- Credentials or secrets committed to the repository, or a Postgres instance
  reachable beyond `127.0.0.1`.
