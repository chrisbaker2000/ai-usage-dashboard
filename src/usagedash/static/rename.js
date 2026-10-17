/**
 * The page's one rename, from opening the form to the server's answer.
 *
 * A rename is in one phase at a time: `idle`, `editing` an account's name, or `saving` it. While a save
 * is in flight nothing else can start, so no older answer can close a newer form, clear a newer failure,
 * or overwrite a newer name. The draft lives here, not in the input, so the page can be rendered again
 * at any time without losing what is being typed.
 */

const IDLE = Object.freeze({ phase: 'idle' });

/**
 * @param {{
 *   save: (accountId: string, label: string) => Promise<{label: string}>,
 *   onChange: () => void,
 *   onFailure: (message: string | null) => void,
 *   onSaved: (accountId: string, label: string) => void,
 * }} dependencies - `save` sends the new name and resolves with the stored one; `onChange` is called on
 *   every phase change; `onFailure` is given the failure to show, or null to clear it; `onSaved` is
 *   called with the stored name before the closed form is rendered.
 * @returns {{
 *   state: () => Readonly<{phase: 'idle'} | {phase: 'editing' | 'saving', accountId: string, draft: string}>,
 *   open: (accountId: string, label: string) => boolean,
 *   type: (text: string) => boolean,
 *   cancel: () => boolean,
 *   submit: () => Promise<boolean>,
 * }} Each action says whether it took effect. `submit` resolves true when the name was saved.
 */
export function createRename({ save, onChange, onFailure, onSaved }) {
  let state = IDLE;
  const enter = (next) => {
    state = Object.freeze(next);
    onChange();
  };
  return {
    state: () => state,

    open(accountId, label) {
      if (state.phase === 'saving') return false;
      onFailure(null);
      enter({ phase: 'editing', accountId, draft: label });
      return true;
    },

    type(text) {
      if (state.phase !== 'editing') return false;
      state = Object.freeze({ ...state, draft: text });
      return true;
    },

    cancel() {
      if (state.phase !== 'editing') return false;
      onFailure(null);
      enter(IDLE);
      return true;
    },

    async submit() {
      if (state.phase !== 'editing') return false;
      const { accountId } = state;
      const label = state.draft.trim();
      if (label === '') return false;
      onFailure(null);
      enter({ phase: 'saving', accountId, draft: label });
      let saved;
      try {
        saved = await save(accountId, label);
      } catch (error) {
        onFailure(`Rename failed: ${error.message}`);
        enter({ phase: 'editing', accountId, draft: label });
        return false;
      }
      state = IDLE;
      onSaved(accountId, saved.label);
      onChange();
      return true;
    },
  };
}

const OUT_OF_QUOTA = 'is out of quota';
const HEADROOM_LEFT = /^has (\d+)% headroom left$/;
const HOLD_TAIL = ' and no other account has comfortable headroom.';
const SWITCH_MARK = '. Switch to ';
const RESET_CLAIM = ' has reset since it was last read, so it may have room. Log in to it to confirm.';
const HAD_MARK = ' had ';
const HAD_TAIL = '% headroom when last read. Log in to it to confirm.';
const STAY_MARK = ' has ';
const STAY_TAIL = '% headroom.';
const SWITCH_TAIL = '% headroom).';
const NO_READING_TAIL = ' has no current reading, so its usage now is unknown.';
const NO_LIMIT_PREFIX = 'No limit that blocks ';
const NO_LIMIT_TAIL = ' has been read yet.';

/**
 * @param {string} value
 * @returns {boolean}
 */
function wholePercent(value) {
  return /^\d+$/.test(value);
}

/**
 * The clause the server puts between an account's name and the rest of a sentence.
 * @param {string} text
 * @returns {{outOfQuota: boolean, percent: string | null} | null}
 */
function constraintParts(text) {
  if (text === OUT_OF_QUOTA) return { outOfQuota: true, percent: null };
  const matched = HEADROOM_LEFT.exec(text);
  if (!matched) return null;
  return { outOfQuota: false, percent: matched[1] };
}

/**
 * @param {{outOfQuota: boolean, percent: string | null}} constraint
 * @returns {string}
 */
function constraintText(constraint) {
  return constraint.outOfQuota ? OUT_OF_QUOTA : `has ${constraint.percent}% headroom left`;
}

/**
 * @param {string} summary
 * @param {string | null} label
 * @returns {string | null} The sentence after `label` and its following space.
 */
function textAfterLabel(summary, label) {
  if (label == null || label === '') return null;
  const prefix = `${label} `;
  if (!summary.startsWith(prefix)) return null;
  return summary.slice(prefix.length);
}

/**
 * Read one server advice sentence back into the account slots it was built from.
 *
 * Why not a search for the old name: a short name is also an ordinary word, two accounts can share a
 * name, and one account's name can be another name plus the words that follow it. The sentence is
 * kept as the server wrote it unless it is one of those sentences and the slots are these accounts.
 * @param {string} summary
 * @param {string | null} activeLabel
 * @param {string | null} targetLabel
 * @returns {object | null}
 */
function parseAdviceSummary(summary, activeLabel, targetLabel) {
  if (activeLabel == null || activeLabel === '') return null;
  if (summary === `${NO_LIMIT_PREFIX}${activeLabel}${NO_LIMIT_TAIL}`) return { kind: 'no-limit' };
  if (summary === `${activeLabel}${NO_READING_TAIL}`) return { kind: 'no-reading' };
  const stay = textAfterLabel(summary, activeLabel);
  if (stay != null && stay.startsWith(STAY_MARK.trimStart()) && summary.endsWith(STAY_TAIL)) {
    const percent = summary.slice(`${activeLabel}${STAY_MARK}`.length, summary.length - STAY_TAIL.length);
    if (summary === `${activeLabel}${STAY_MARK}${percent}${STAY_TAIL}` && wholePercent(percent)) {
      return { kind: 'stay', percent };
    }
  }
  const rest = textAfterLabel(summary, activeLabel);
  if (rest == null) return null;
  if (rest.endsWith(HOLD_TAIL)) {
    const constraint = constraintParts(rest.slice(0, rest.length - HOLD_TAIL.length));
    if (constraint) return { kind: 'hold', constraint };
  }
  const switchAt = rest.indexOf(SWITCH_MARK);
  if (switchAt !== -1 && targetLabel != null && targetLabel !== '') {
    const constraint = constraintParts(rest.slice(0, switchAt));
    const after = rest.slice(switchAt + SWITCH_MARK.length);
    const opened = `${targetLabel} (`;
    if (constraint && after.startsWith(opened) && after.endsWith(SWITCH_TAIL)) {
      const percent = after.slice(opened.length, after.length - SWITCH_TAIL.length);
      if (wholePercent(percent)) return { kind: 'switch', constraint, percent };
    }
  }
  const claimAt = rest.indexOf('. ');
  if (claimAt !== -1 && targetLabel != null && targetLabel !== '') {
    const constraint = constraintParts(rest.slice(0, claimAt));
    const claim = rest.slice(claimAt + 2);
    if (constraint && claim === `${targetLabel}${RESET_CLAIM}`) return { kind: 'verify-reset', constraint };
    const had = `${targetLabel}${HAD_MARK}`;
    if (constraint && claim.startsWith(had) && claim.endsWith(HAD_TAIL)) {
      const percent = claim.slice(had.length, claim.length - HAD_TAIL.length);
      if (wholePercent(percent)) return { kind: 'verify-had', constraint, percent };
    }
  }
  return null;
}

/**
 * @param {object} parsed
 * @param {string | null} activeLabel
 * @param {string | null} targetLabel
 * @returns {string}
 */
function formatAdviceSummary(parsed, activeLabel, targetLabel) {
  if (activeLabel == null) return '';
  if (parsed.kind === 'no-limit') return `${NO_LIMIT_PREFIX}${activeLabel}${NO_LIMIT_TAIL}`;
  if (parsed.kind === 'no-reading') return `${activeLabel}${NO_READING_TAIL}`;
  if (parsed.kind === 'stay') return `${activeLabel}${STAY_MARK}${parsed.percent}${STAY_TAIL}`;
  const constraint = constraintText(parsed.constraint);
  if (parsed.kind === 'hold') return `${activeLabel} ${constraint}${HOLD_TAIL}`;
  if (targetLabel == null) return '';
  if (parsed.kind === 'switch') {
    return `${activeLabel} ${constraint}${SWITCH_MARK}${targetLabel} (${parsed.percent}% headroom).`;
  }
  if (parsed.kind === 'verify-reset') return `${activeLabel} ${constraint}. ${targetLabel}${RESET_CLAIM}`;
  return `${activeLabel} ${constraint}. ${targetLabel}${HAD_MARK}${parsed.percent}${HAD_TAIL}`;
}

/**
 * @param {{summary: string, active_account_id?: string | null, target_account_id?: string | null}} advice
 * @param {Array<{id: string, label: string}>} accounts
 * @param {string} accountId
 * @param {string} label
 * @returns {string}
 */
function summaryWithRenamedAccount(advice, accounts, accountId, label) {
  const current = accounts.find((account) => account.id === accountId);
  if (!current || current.label === '' || current.label === label) return advice.summary;
  const active = accounts.find((account) => account.id === advice.active_account_id) ?? null;
  const target = accounts.find((account) => account.id === advice.target_account_id) ?? null;
  const parsed = parseAdviceSummary(advice.summary, active?.label ?? null, target?.label ?? null);
  if (parsed == null) return advice.summary;
  const activeLabel = active != null && active.id === accountId ? label : (active?.label ?? null);
  const targetLabel = target != null && target.id === accountId ? label : (target?.label ?? null);
  const rewritten = formatAdviceSummary(parsed, activeLabel, targetLabel);
  return rewritten === '' ? advice.summary : rewritten;
}

/**
 * Returns a dashboard state with one account's label replaced and everything else as it was, so a saved
 * name is shown at once. An advice sentence is rewritten only when it is one the server builds and this
 * account is the active or the recommended account in it.
 *
 * @param {{providers: Array<{accounts: Array<{id: string, label: string}>, advice?: {summary: string, active_account_id?: string | null, target_account_id?: string | null}}>}} dashboardState
 * @param {string} accountId - The renamed account. Ids are decimal text, exact however large.
 * @param {string} label - Its new name.
 * @returns {object} A new state; the one given is not changed.
 */
export function withAccountLabel(dashboardState, accountId, label) {
  return {
    ...dashboardState,
    providers: dashboardState.providers.map((provider) => {
      const current = provider.accounts.find((account) => account.id === accountId);
      if (!current) return provider;
      return {
        ...provider,
        accounts: provider.accounts.map((account) => (account.id === accountId ? { ...account, label } : account)),
        advice: provider.advice
          ? {
              ...provider.advice,
              summary: summaryWithRenamedAccount(provider.advice, provider.accounts, accountId, label),
            }
          : provider.advice,
      };
    }),
  };
}

/**
 * Returns a token report with one account's name replaced. The report is separate from the quota
 * state, and a failed reload would otherwise keep showing the old name beside the new one.
 *
 * @param {{series: Array<{account_id: string, label: string}>} | null} report
 * @param {string} accountId
 * @param {string} label
 * @returns {object | null} A new report, or null when there is none yet. The one given is not changed.
 */
export function withTokenAccountLabel(report, accountId, label) {
  if (report == null) return report;
  return {
    ...report,
    series: report.series.map((series) => (series.account_id === accountId ? { ...series, label } : series)),
  };
}
