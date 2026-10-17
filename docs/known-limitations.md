# Known limitations

Install, everyday use, and the meaning of the page are in the [README](../README.md). This file
records behaviour that is deliberate and understood, so it is not taken for a defect or fixed in a
way that was already tried.

- [A Claude response first seen after part of it has expired](#a-claude-response-first-seen-after-part-of-it-has-expired)
- [A Codex map entry is kept when the main pool names no meter](#a-codex-map-entry-is-kept-when-the-main-pool-names-no-meter)
- [The access token is created once and stays valid](#the-access-token-is-created-once-and-stays-valid)
- [The retail API cost uses the short-prompt standard rate](#the-retail-api-cost-uses-the-short-prompt-standard-rate)

## A Claude response first seen after part of it has expired

A Claude response is counted once, at the largest value of each count, under its earliest line
(`db.add_message_usage`). Once its usage expires (`db.prune`, 400 days), its key is kept in
`claude_expired_messages` and the response is never counted again, whichever of its lines a later
scan reads.

A response the store has never seen, and whose lines fall on both sides of the start of the first
day retention keeps, is counted from whichever of its lines arrive first:

- its line inside retention, read alone, counts it under that line's hour, inside what is kept;
- its earlier line, read with it or later, moves it into the expired hours, and the next prune
  removes it and remembers its key.

So the totals for that one response depend on which of its lines are read, and in what order, until
the earlier one is read. Only a transcript holding usage more than 400 days old that the dashboard
never read before can do this: a session resumed after that long copies its history into a new
transcript. A response's lines are written while it streams, minutes apart, so it straddles the
start of a day only if it was streaming at midnight UTC.

Refusing any response first seen within a day after the first kept day was tried and withdrawn. It
moved the same dependence on read order to the end of that day, and a refused response came back
when a later scan read only its later line.

## A Codex map entry is kept when the main pool names no meter

The meter map is keyed by meter id, and that id is the main pool's `limitId`
(`codex_cli.parse_rate_limits`). When the main pool names one, the map entry with that id is the
same pool and is not shown again.

When the main pool names none, the same windows, reached flag, and display name are not a meter
id. Two meters can both be an unnamed weekly window at the same percentage. Skipping every entry
that matches would drop those meters and record nothing about them. Those entries are kept, so a
copy of the main pool can also appear once more, as a limit that does not block.

Treating a matching snapshot as the main pool was tried and withdrawn. Distinct meters with that
snapshot disappeared.

## The access token is created once and stays valid

`ensure_access_token` creates the token by linking a finished file into place, and every later caller,
the server and `usagedash open` alike, returns that same token. Replacing it on every start would sign
the open browser out on every launchd restart.

The dashboard therefore never has to receive the token. The page is an owner-only local file that
holds the token as a JavaScript value. Before each request it asks the process it is talking to for a
challenge (`GET /challenge`), adds a nonce of its own, and signs method, path and query, body digest,
challenge, and nonce with the token (`Authorization: Usagedash <challenge>.<nonce>.<digest>`,
`access.request_signature`). The server accepts a signature only over a challenge it issued and has not
seen used (`access.RequestVerifier`), and signs its answer, whatever it is, over the challenge, the nonce,
the status, and the body (`X-Usagedash-Response`); the page uses an answer only when that signature is
its own. The server's challenge keeps requests fresh and the page's nonce keeps answers fresh. Every body
is sealed as well (`access.BodySealer`, AES-256-GCM under a key both sides derive from the token) and
bound to the same exchange; a signed request whose body is in clear or does not open is refused
(`api.open_signed_body`). A process that has only just bound the vacated loopback port receives
signatures over challenges of its own, never the token, and sees sealed bytes: it cannot sign, alter,
replay, or read anything. Nothing depends on the wall clock. The page addresses `127.0.0.1` only, never
`localhost`, which a browser may resolve to `::1`.

What remains, and why it is accepted:

- **Each request is two round trips, and a restart between them fails one poll**: the new process does
  not know the challenge; the next poll, 30 seconds later, succeeds.
- **Challenges can be asked for without a token.** At most 256 are held and the oldest is forgotten
  first, so a flood only makes the page's request fail until its next poll; a signature that does not
  match leaves its challenge held.
- **A token an earlier page stored in the loopback origin stays readable there until the dashboard's own
  page runs** (`access.forgetStoredToken`), or until that origin's site data is cleared.
- **A bookmark or history entry saved from the earlier hand-off still holds the token in its fragment.**
  It signs nothing in, but holds the current token until the token file is deleted and recreated.
- **The page's code is copied into the file** (`access.assemble_bootstrap_page`): a browser refuses a
  module loaded from a `file:` URL, and one served by the listener would run with the token in reach. A
  page opened before an update runs the older page until `usagedash open` is run again.

These designs were tried and withdrawn:

- The token in a URL fragment after the listener signed a nonce. A process that had just bound the
  port received it.
- Signing over a page nonce and the wall-clock time. A restart or a clock set back let a captured
  signature replay once.
- Signing without sealing. A port holder read a rename's label, and one relaying a kept connection
  read both directions.
- Freshness from the listener's challenge alone. A recorded answer replayed behind a re-issued
  challenge.
- A tokenless served shell polling unsigned. A port holder could feed it fabricated data.

## The retail API cost uses the short-prompt standard rate

The token section prices each account's tokens at the vendors' published pay-as-you-go rates
(`pricing.py`, tables read 2026-10-08). The figure is what those tokens would have cost on the API.
It is not the subscription charge.

An hourly total does not say which requests were long-context, batch, fast, priority, or served from
a regional endpoint, so those multipliers are not applied. That includes Claude's higher price above
the short prompt, the US inference multiplier on Claude 4.6 and later, OpenAI's data-residency and
FedRAMP percentages, and xAI's US-endpoint and priority multipliers.

Claude stores cache writes as one count. The price uses the published 5-minute write rate (1.25 times
input), not the 1-hour rate (2 times input). Haiku 5.5 is priced by prompt length; the figure uses
the rate at or below 100,000 prompt tokens.

`grok-4.7-build` and `grok-4.6-build` are priced as the public short-context `grok-4.7` and
`grok-4.6` models. `grok-4.7-fast` is a different product and is not priced. xAI publishes no
cache-write rate, so a Grok model that recorded cache writes is left out of the dollar total.

`gpt-5.6-sol` uses the published standard rate. That rate is promotional at least through
21 November 2026.

The page rounds each shown figure to the nearest dollar. The all-accounts row adds those rounded
figures, so it can differ by a dollar from rounding the exact combined total. Half a dollar rounds
away from zero.

Reasoning tokens are part of output and are not priced a second time. A model with no published rate
for a component that was used stays in the table with its token count and is left out of the dollar
total. No partial price is shown for that model.

This is the comparable API price hourly totals can support. They do not carry the request-level
qualifiers the vendors bill on.
