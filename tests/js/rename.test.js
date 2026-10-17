import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import { createRename, withAccountLabel, withTokenAccountLabel } from '../../src/usagedash/static/rename.js';

/** A rename whose saves are answered by the test, with everything it was told recorded. */
function harness() {
  const saves = [];
  const events = [];
  const rename = createRename({
    save: (accountId, label) =>
      new Promise((resolve, reject) => {
        saves.push({ accountId, label, resolve, reject });
      }),
    onChange: () => events.push(`render ${rename.state().phase}`),
    onFailure: (message) => events.push(message === null ? 'no failure' : message),
    onSaved: (accountId, label) => events.push(`saved ${accountId} as ${label}`),
  });
  return { rename, saves, events };
}

describe('editing a name', () => {
  it('starts idle', () => {
    assert.deepEqual(harness().rename.state(), { phase: 'idle' });
  });

  it('opens with the current name as the draft and asks the page to render', () => {
    const { rename, events } = harness();

    assert.equal(rename.open('7', 'Account A'), true);

    assert.deepEqual(rename.state(), { phase: 'editing', accountId: '7', draft: 'Account A' });
    assert.deepEqual(events, ['no failure', 'render editing']);
  });

  it('keeps what is typed without asking the page to render on every keystroke', () => {
    const { rename, events } = harness();
    rename.open('7', 'Account A');
    events.length = 0;

    assert.equal(rename.type('Personal'), true);

    assert.equal(rename.state().draft, 'Personal');
    assert.deepEqual(events, []);
  });

  it('replaces an open edit and its draft when another account is opened', () => {
    const { rename } = harness();
    rename.open('7', 'Account A');
    rename.type('half-typed');

    assert.equal(rename.open('8', 'Account B'), true);

    assert.deepEqual(rename.state(), { phase: 'editing', accountId: '8', draft: 'Account B' });
  });

  it('closes on cancel', () => {
    const { rename, events } = harness();
    rename.open('7', 'Account A');
    events.length = 0;

    assert.equal(rename.cancel(), true);

    assert.deepEqual(rename.state(), { phase: 'idle' });
    assert.deepEqual(events, ['no failure', 'render idle']);
  });

  it('has nothing to type into, cancel, or submit when idle', async () => {
    const { rename, saves, events } = harness();

    assert.equal(rename.type('text'), false);
    assert.equal(rename.cancel(), false);
    assert.equal(await rename.submit(), false);

    assert.deepEqual(rename.state(), { phase: 'idle' });
    assert.deepEqual([saves, events], [[], []]);
  });

  it('does not let the state it hands out be changed', () => {
    const { rename } = harness();
    rename.open('7', 'Account A');

    assert.throws(() => {
      rename.state().draft = 'changed from outside';
    }, TypeError);
  });
});

describe('saving a name', () => {
  it('sends the trimmed draft, freezes the form while it is in flight, and closes it when saved', async () => {
    const { rename, saves, events } = harness();
    rename.open('7', 'Account A');
    rename.type('  Personal  ');
    events.length = 0;

    const submitted = rename.submit();

    assert.deepEqual(rename.state(), { phase: 'saving', accountId: '7', draft: 'Personal' });
    assert.deepEqual(
      saves.map(({ accountId, label }) => [accountId, label]),
      [['7', 'Personal']],
    );
    saves[0].resolve({ label: 'Personal' });
    assert.equal(await submitted, true);
    assert.deepEqual(rename.state(), { phase: 'idle' });
    assert.deepEqual(events, ['no failure', 'render saving', 'saved 7 as Personal', 'render idle']);
  });

  it('tells the page the name the server stored, before asking it to render the closed form', async () => {
    const { rename, saves, events } = harness();
    rename.open('7', 'Account A');
    rename.type('Personal');
    const submitted = rename.submit();

    saves[0].resolve({ label: 'Personal (as stored)' });
    await submitted;

    assert.deepEqual(events.slice(-2), ['saved 7 as Personal (as stored)', 'render idle']);
  });

  for (const blank of ['', '   ', '\t\n']) {
    it(`does not send a blank name (${JSON.stringify(blank)})`, async () => {
      const { rename, saves } = harness();
      rename.open('7', 'Account A');
      rename.type(blank);

      assert.equal(await rename.submit(), false);

      assert.deepEqual(saves, []);
      assert.equal(rename.state().phase, 'editing');
    });
  }

  it('reopens the form with the draft and reports the failure when the save fails', async () => {
    const { rename, saves, events } = harness();
    rename.open('7', 'Account A');
    rename.type('Personal');
    events.length = 0;
    const submitted = rename.submit();

    saves[0].reject(new Error('/api/accounts/7 responded 500'));

    assert.equal(await submitted, false);
    assert.deepEqual(rename.state(), { phase: 'editing', accountId: '7', draft: 'Personal' });
    assert.deepEqual(events, [
      'no failure',
      'render saving',
      'Rename failed: /api/accounts/7 responded 500',
      'render editing',
    ]);
  });

  it('clears an earlier failure when the save is tried again, and when the form is cancelled', async () => {
    const { rename, saves, events } = harness();
    rename.open('7', 'Account A');
    const failed = rename.submit();
    saves[0].reject(new Error('offline'));
    await failed;
    events.length = 0;

    const retried = rename.submit();
    saves[1].resolve({ label: 'Account A' });
    await retried;

    assert.equal(events[0], 'no failure');
    assert.equal(events.includes('Rename failed: offline'), false);

    rename.open('7', 'Account A');
    events.length = 0;
    rename.cancel();
    assert.deepEqual(events, ['no failure', 'render idle']);
  });
});

describe('while a save is in flight', () => {
  function saving() {
    const context = harness();
    context.rename.open('7', 'Account A');
    context.rename.type('Personal');
    context.submitted = context.rename.submit();
    context.events.length = 0;
    return context;
  }

  it('ignores a second submit, so one form cannot have two saves under way', async () => {
    const { rename, saves, submitted } = saving();

    assert.equal(await rename.submit(), false);

    assert.equal(saves.length, 1);
    saves[0].resolve({ label: 'Personal' });
    assert.equal(await submitted, true);
  });

  it('ignores typing, so a draft entered after submitting cannot be discarded by the answer', async () => {
    const { rename, saves, submitted } = saving();

    assert.equal(rename.type('typed after submitting'), false);

    assert.equal(rename.state().draft, 'Personal');
    saves[0].reject(new Error('offline'));
    await submitted;
    assert.deepEqual(rename.state(), { phase: 'editing', accountId: '7', draft: 'Personal' });
  });

  it('cannot be cancelled, so the same account cannot be reopened and saved again behind it', async () => {
    const { rename, saves, events, submitted } = saving();

    assert.equal(rename.cancel(), false);
    assert.equal(rename.open('7', 'Account A'), false);

    assert.equal(rename.state().phase, 'saving');
    assert.deepEqual(events, []);
    saves[0].resolve({ label: 'Personal' });
    await submitted;
  });

  it('cannot be replaced by an edit of another account', async () => {
    const { rename, saves, submitted } = saving();

    assert.equal(rename.open('8', 'Account B'), false);

    assert.deepEqual(rename.state(), { phase: 'saving', accountId: '7', draft: 'Personal' });
    saves[0].resolve({ label: 'Personal' });
    await submitted;
    assert.equal(rename.open('8', 'Account B'), true);
  });

  it('sends saves one after another in the order they were submitted', async () => {
    const { rename, saves, submitted } = saving();
    saves[0].resolve({ label: 'Personal' });
    await submitted;

    rename.open('7', 'Personal');
    rename.type('Personal 2');
    const second = rename.submit();
    assert.deepEqual(
      saves.map(({ label }) => label),
      ['Personal', 'Personal 2'],
    );
    saves[1].resolve({ label: 'Personal 2' });
    assert.equal(await second, true);
  });
});

function adviceState(summary, accounts) {
  const target = accounts.find((account) => account.id !== '7') ?? null;
  return {
    providers: [
      {
        provider: 'claude',
        advice: {
          summary,
          active_account_id: '7',
          target_account_id: target == null ? null : target.id,
        },
        accounts,
      },
      {
        provider: 'codex',
        advice: { summary: 'Ann has room.' },
        accounts: [{ id: '9', label: 'Ann' }],
      },
    ],
  };
}

describe('showing a saved name at once', () => {
  const dashboardState = {
    generated_at: '2026-10-03T14:00:00+00:00',
    providers: [
      {
        provider: 'claude',
        accounts: [
          { id: '7', label: 'Account A' },
          { id: '8', label: 'Account B' },
        ],
      },
      { provider: 'codex', accounts: [{ id: '9', label: 'Account C' }] },
    ],
  };

  it('replaces the label of that account only', () => {
    const renamed = withAccountLabel(dashboardState, '8', 'Work');

    assert.deepEqual(
      renamed.providers.map((provider) => provider.accounts.map((account) => account.label)),
      [['Account A', 'Work'], ['Account C']],
    );
    assert.equal(renamed.generated_at, dashboardState.generated_at);
  });

  it('leaves the state it was given unchanged', () => {
    withAccountLabel(dashboardState, '8', 'Work');

    assert.equal(dashboardState.providers[0].accounts[1].label, 'Account B');
  });

  it('changes nothing when no account has that id', () => {
    assert.deepEqual(withAccountLabel(dashboardState, '99', 'Nobody'), dashboardState);
  });

  it('tells apart two ids that are the same number to a browser', () => {
    const neighbours = {
      providers: [
        {
          provider: 'claude',
          accounts: [
            { id: '9007199254740992', label: 'Account A' },
            { id: '9007199254740993', label: 'Account B' },
          ],
        },
      ],
    };
    assert.equal(Number(neighbours.providers[0].accounts[0].id), Number(neighbours.providers[0].accounts[1].id));

    const renamed = withAccountLabel(neighbours, '9007199254740993', 'Work');

    assert.deepEqual(
      renamed.providers[0].accounts.map((account) => account.label),
      ['Account A', 'Work'],
    );
  });

  it('renames the active account in a switch sentence and leaves the other account', () => {
    const state = adviceState('Ann has 10% headroom left. Switch to Anne (60% headroom).', [
      { id: '7', label: 'Ann' },
      { id: '8', label: 'Anne' },
    ]);

    const renamed = withAccountLabel(state, '7', 'Work');

    assert.equal(renamed.providers[0].advice.summary, 'Work has 10% headroom left. Switch to Anne (60% headroom).');
    assert.equal(renamed.providers[1].advice.summary, 'Ann has room.');
    assert.equal(state.providers[0].advice.summary, 'Ann has 10% headroom left. Switch to Anne (60% headroom).');
  });

  it('renames only the active account when both accounts have the same name', () => {
    const state = adviceState('Personal has 10% headroom left. Switch to Personal (60% headroom).', [
      { id: '7', label: 'Personal' },
      { id: '8', label: 'Personal' },
    ]);

    const renamed = withAccountLabel(state, '7', 'Work');

    assert.equal(renamed.providers[0].advice.summary, 'Work has 10% headroom left. Switch to Personal (60% headroom).');
    assert.deepEqual(
      renamed.providers[0].accounts.map((account) => account.label),
      ['Work', 'Personal'],
    );
  });

  it('renames the recommended account and leaves the active name', () => {
    const state = adviceState('Ann is out of quota. Switch to Anne (60% headroom).', [
      { id: '7', label: 'Ann' },
      { id: '8', label: 'Anne' },
    ]);

    assert.equal(
      withAccountLabel(state, '8', 'Work').providers[0].advice.summary,
      'Ann is out of quota. Switch to Work (60% headroom).',
    );
  });

  it('does not rewrite ordinary words when the name is a single letter', () => {
    const state = adviceState('a has 10% headroom.', [
      { id: '7', label: 'a' },
      { id: '8', label: 'has' },
    ]);

    assert.equal(withAccountLabel(state, '7', 'Work').providers[0].advice.summary, 'Work has 10% headroom.');
    assert.equal(withAccountLabel(state, '8', 'Room').providers[0].advice.summary, 'a has 10% headroom.');
  });

  it('renames Ann when another account is labelled with Ann plus the following words', () => {
    const state = adviceState('Ann has 90% headroom.', [
      { id: '7', label: 'Ann' },
      { id: '8', label: 'Ann has' },
    ]);

    assert.equal(withAccountLabel(state, '7', 'Work').providers[0].advice.summary, 'Work has 90% headroom.');
  });

  it('leaves a sentence that is not one the server builds', () => {
    const state = adviceState('Ann has room.', [{ id: '7', label: 'Ann' }]);

    assert.equal(withAccountLabel(state, '7', 'Work').providers[0].advice.summary, 'Ann has room.');
  });

  it('renames the account inside each other server sentence', () => {
    const cases = [
      ['No limit that blocks Ann has been read yet.', 'No limit that blocks Work has been read yet.', '7'],
      [
        'Ann has no current reading, so its usage now is unknown.',
        'Work has no current reading, so its usage now is unknown.',
        '7',
      ],
      [
        'Ann is out of quota and no other account has comfortable headroom.',
        'Work is out of quota and no other account has comfortable headroom.',
        '7',
      ],
      [
        'Ann has 10% headroom left. Anne has reset since it was last read, so it may have room. Log in to it to confirm.',
        'Work has 10% headroom left. Anne has reset since it was last read, so it may have room. Log in to it to confirm.',
        '7',
      ],
      [
        'Ann is out of quota. Anne had 60% headroom when last read. Log in to it to confirm.',
        'Ann is out of quota. Work had 60% headroom when last read. Log in to it to confirm.',
        '8',
      ],
    ];

    for (const [summary, expected, renamedId] of cases) {
      const accounts = [
        { id: '7', label: 'Ann' },
        { id: '8', label: 'Anne' },
      ];
      const state = adviceState(summary, accounts);
      assert.equal(withAccountLabel(state, renamedId, 'Work').providers[0].advice.summary, expected);
    }
  });

  it('replaces that account in the token report and leaves the others', () => {
    const report = {
      series: [
        { account_id: '7', label: 'Ann' },
        { account_id: '8', label: 'Anne' },
      ],
    };

    const renamed = withTokenAccountLabel(report, '7', 'Work');

    assert.deepEqual(
      renamed.series.map((series) => series.label),
      ['Work', 'Anne'],
    );
    assert.equal(report.series[0].label, 'Ann');
    assert.equal(withTokenAccountLabel(null, '7', 'Work'), null);
  });
});
