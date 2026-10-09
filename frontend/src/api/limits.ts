/**
 * How long a sentence and a card's phrase may be: the server's own numbers (the owner, 2026-10-08). And how many rules
 * a sort takes beside its first (the owner, 2026-10-09).
 *
 * The fields stop typing here, and a counter shows the room left once a field is 80 % full, so a long description is
 * never refused at Start with "Die Anfrage ist ungültig." (the morning of 2026-10-08: 85 characters into the card's
 * 80). `tests/test_api_text_limits.py` reads these two numbers and compares them with the server's: the reader's
 * `MAX_SENTENCE_CHARS` (`CommandIn.text`, `CommandProvenanceIn.text`) and `MAX_FIELD_CHARS` (`TaskIn.object`,
 * `CameraPlaceIn.phrase`). A text over its limit is refused anyway, 422 `text_too_long` with the limit.
 */

/** The longest sentence the reader takes, typed or spoken. */
export const MAX_SENTENCE_CHARS = 1000

/** The longest phrase a card's field takes: what to pick, what the camera looks for. */
export const MAX_FIELD_CHARS = 200

/** The most rules a sort takes beside its first, the task's own fields (`TaskIn.more_rules`, the server's
 *  `MAX_FURTHER_RULES`): "Grüne Teile in die gelbe Kiste, rote in die blaue" is two, four in all at most. */
export const MAX_MORE_RULES = 3

/** From this share of its limit on, a field shows how many characters it holds. */
export const COUNT_FROM = 0.8

/** Whether a field holding `length` of `limit` characters shows its counter. */
export function showsCount(length: number, limit: number): boolean {
  return length >= Math.floor(limit * COUNT_FROM)
}
