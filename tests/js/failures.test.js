import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { createFailureBoard, recordingOutcome } from '../../src/usagedash/static/failures.js';
import { latestOnly } from '../../src/usagedash/static/latest.js';

/** A load whose requests are answered by the test, in whatever order it chooses. */
function controllableLoad() {
  const pending = [];
  const load = () => new Promise((resolve, reject) => pending.push({ resolve, reject }));
  return { load, pending };
}

/** A board that keeps every list of messages it was asked to show. */
function recordedBoard() {
  const rendered = [];
  const board = createFailureBoard((messages) => rendered.push(messages));
  return { board, rendered, showing: () => rendered.at(-1) ?? [] };
}

const describeFailure = (error) => `Could not load: ${error.message}`;

describe('one failure per source', () => {
  it('shows a reported failure', () => {
    const { board, showing } = recordedBoard();

    board.report('state', 'state is unavailable');

    assert.deepEqual(showing(), ['state is unavailable']);
  });

  it('keeps one source failing when another source succeeds', () => {
    const { board, showing } = recordedBoard();
    board.report('tokens', 'tokens are unavailable');

    board.clear('state');

    assert.deepEqual(showing(), ['tokens are unavailable']);
  });

  it('removes a failure only when its own source clears it', () => {
    const { board, showing } = recordedBoard();
    board.report('state', 'state is unavailable');
    board.report('tokens', 'tokens are unavailable');

    board.clear('state');

    assert.deepEqual(showing(), ['tokens are unavailable']);
  });

  it("replaces a source's earlier failure with its newer one", () => {
    const { board, showing } = recordedBoard();
    board.report('state', 'responded 500');

    board.report('state', 'responded 503');

    assert.deepEqual(showing(), ['responded 503']);
  });

  it('shows a message two sources share once, and keeps it until both have cleared', () => {
    const { board, showing } = recordedBoard();
    board.report('state', 'not signed in');
    board.report('tokens', 'not signed in');
    assert.deepEqual(showing(), ['not signed in']);

    board.clear('state');
    assert.deepEqual(showing(), ['not signed in']);

    board.clear('tokens');
    assert.deepEqual(showing(), []);
  });

  it('shows failures oldest first', () => {
    const { board, showing } = recordedBoard();
    board.report('refresh', 'refresh failed');
    board.report('state', 'state is unavailable');

    assert.deepEqual(showing(), ['refresh failed', 'state is unavailable']);
  });

  it('does not render again when a source with no failure clears', () => {
    const { board, rendered } = recordedBoard();

    board.clear('state');

    assert.deepEqual(rendered, []);
  });
});

describe('recording the outcome of a load', () => {
  it('reports the failure of the newest call and resolves false', async () => {
    const { board, showing } = recordedBoard();
    const run = recordingOutcome(() => Promise.reject(new Error('offline')), board, 'state', describeFailure);

    assert.equal(await run(), false);
    assert.deepEqual(showing(), ['Could not load: offline']);
  });

  it('clears its own failure when a later call is shown', async () => {
    const { board, showing } = recordedBoard();
    const answers = [Promise.reject(new Error('offline')), Promise.resolve(true)];
    const run = recordingOutcome(() => answers.shift(), board, 'state', describeFailure);
    await run();

    assert.equal(await run(), true);
    assert.deepEqual(showing(), []);
  });

  it('leaves the failures alone when its call was superseded', async () => {
    const { board, rendered } = recordedBoard();
    board.report('state', 'Could not load: offline');
    const run = recordingOutcome(() => Promise.resolve(false), board, 'state', describeFailure);

    assert.equal(await run(), false);
    assert.deepEqual(rendered, [['Could not load: offline']]);
  });

  it("does not clear another source's failure when it succeeds", async () => {
    const { board, showing } = recordedBoard();
    board.report('tokens', 'Could not load: /api/tokens responded 500');
    const run = recordingOutcome(() => Promise.resolve(true), board, 'state', describeFailure);

    await run();

    assert.deepEqual(showing(), ['Could not load: /api/tokens responded 500']);
  });
});

describe('overlapping loads that fail', () => {
  it("keeps the newest call's failure when an older call is answered after it", async () => {
    const { load, pending } = controllableLoad();
    const shown = [];
    const { board, showing } = recordedBoard();
    const run = recordingOutcome(
      latestOnly(load, (result) => shown.push(result)),
      board,
      'tokens',
      describeFailure,
    );

    const older = run();
    const newer = run();
    pending[1].reject(new Error('offline'));
    await newer;
    pending[0].resolve('30d report');

    assert.deepEqual(await Promise.all([older, newer]), [false, false]);
    assert.deepEqual(showing(), ['Could not load: offline']);
    assert.deepEqual(shown, []);
  });

  it('does not report the failure of an older call that a newer call replaced', async () => {
    const { load, pending } = controllableLoad();
    const shown = [];
    const { board, rendered } = recordedBoard();
    const run = recordingOutcome(
      latestOnly(load, (result) => shown.push(result)),
      board,
      'tokens',
      describeFailure,
    );

    const older = run();
    const newer = run();
    pending[0].reject(new Error('offline'));
    pending[1].resolve('24h report');

    assert.deepEqual(await Promise.all([older, newer]), [false, true]);
    assert.deepEqual(rendered, []);
    assert.deepEqual(shown, ['24h report']);
  });
});
