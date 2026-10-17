# Security

## Reporting a vulnerability

Open a [private security advisory](https://github.com/chrisbaker2000/ai-usage-dashboard/security/advisories/new)
on this repository. Do not file a public issue for a vulnerability, and do not include tokens,
session files, keychain contents, or account identifiers in the report.

## What this program is

The dashboard runs as you, on your Mac, and listens on `127.0.0.1` only. It reads a vendor CLI's
credential so it can ask that CLI for quota. It does not copy a working credential into its own
store. A standing login lives in its own home under the data directory, and that CLI refreshes its
own session.

`usagedash open` writes `open-dashboard.html`. That file holds the access token the browser on this
Mac uses to talk to the local server. The file is owner-only. Do not copy it to another machine or
another account.

A config file, data directory, standing home, or vendor executable that another macOS account could
change is refused. The design of the page token is in
[docs/known-limitations.md](docs/known-limitations.md).

There is no hosted service. A reading stays on this Mac, apart from the requests the vendor CLI you
installed already makes.
