/**
 * Wraps an asynchronous load so that only its newest call takes effect: a superseded call's answer, data
 * or error, describes a request the page no longer wants and may not replace what the newer call shows.
 *
 * @template T
 * @param {() => Promise<T>} load - Starts the request.
 * @param {(result: T) => void} show - Applies a result; called only for the newest call.
 * @returns {() => Promise<boolean>} Runs one load: true when its result was shown, false when a newer call
 *   superseded it, rejecting with the load's error only when the failed call was still the newest.
 */
export function latestOnly(load, show) {
  let newest = 0;
  return async () => {
    newest += 1;
    const ticket = newest;
    let result;
    try {
      result = await load();
    } catch (error) {
      if (ticket === newest) throw error;
      return false;
    }
    if (ticket !== newest) return false;
    show(result);
    return true;
  };
}
