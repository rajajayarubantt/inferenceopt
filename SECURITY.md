# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository (Security tab → "Report a vulnerability"), or by email to
rajajayarubant@gmail.com.

Include the affected version, a description, reproduction steps and the impact you
expect. We aim to acknowledge reports within 5 working days.

## Supported versions

Until 1.0, only the latest released `0.x` version receives security fixes.

## Scope notes

InferenceOpt starts inference runtimes and can run benchmark workloads on the host. The
web UI is designed for **local, single-user** use:

- It binds to `127.0.0.1` by default and refuses other addresses without `--allow-remote`.
- Every API call requires the per-launch session token.
- Host-header and Origin checks block DNS-rebinding and cross-site requests.

Reports about bypassing any of these protections are in scope. Running the UI on a
network without TLS and a reverse proxy is not a supported configuration.
