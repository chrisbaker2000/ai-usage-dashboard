/**
 * The failures the page is currently showing, one per source.
 *
 * Each source (a load, an action) reports and clears only its own failure. One source succeeding
 * therefore never hides another's failure, and a failure stays until that same source succeeds.
 */

/**
 * @param {(messages: string[]) => void} render - Shows the current failures. Called on every change with
 *   each distinct message once, oldest first.
 * @returns {{report: (source: string, message: string) => void, clear: (source: string) => void}}
 */
export function createFailureBoard(render) {
  const failures = new Map();
  const publish = () => render([...new Set(failures.values())]);
  return {
    report(source, message) {
      failures.set(source, message);
      publish();
    },
    clear(source) {
      if (failures.delete(source)) publish();
    },
  };
}

/**
 * Records a latest-only load's outcome against its own source.
 *
 * `load` keeps the `latestOnly` contract: true when its result was shown, false when a newer call
 * superseded it, rejecting only when the failed call was still the newest. A superseded call records
 * nothing; the call that replaced it decides what is shown and reported.
 *
 * @param {() => Promise<boolean>} load - Runs one load under the `latestOnly` contract.
 * @param {{report: (source: string, message: string) => void, clear: (source: string) => void}} board
 * @param {string} source - The name this load's failure is kept under.
 * @param {(error: Error) => string} describe - Turns a failure into the message to show.
 * @returns {() => Promise<boolean>} Runs one load; resolves true when its result was shown, never rejects.
 */
export function recordingOutcome(load, board, source, describe) {
  return async () => {
    let wasShown;
    try {
      wasShown = await load();
    } catch (error) {
      board.report(source, describe(error));
      return false;
    }
    if (wasShown) board.clear(source);
    return wasShown;
  };
}
