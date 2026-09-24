# Tax-Planning

A reference guide of U.S. tax-planning strategies, deployed live via Cloudflare Workers from this repo's `main` branch.

## Layout

| Path | What it is |
|---|---|
| `guides/index.html` | The guide: every strategy, table of contents, search, category filter, progress tracking, flags and the chat panel, in one page. |
| `guides/Tax_Strategies.html` | An identical copy of `index.html`, kept for the older URL. |
| `guides/Tax_Strategies.docx` | Word version of the guide. |
| `guides/404.html` | Not-found page. |
| `src/strategies-data.generated.js` | The strategy content as data (title, references, intro, sections, categories, plain-terms summary), used by the chat Worker. |
| `src/worker.js` | Cloudflare Worker: the `/api/*` routes (chat, progress, flags, embeddings admin); everything else is served from `guides/`. |
| `wrangler.jsonc` | Worker, static-assets and KV configuration. |

## Editing a strategy

Each strategy's text lives in several places that must stay identical:

1. The visible `<article class="strategy">` in `guides/index.html`.
2. Its entry in `src/strategies-data.generated.js` (the references list lives only here).
3. The `data-search` attribute on both the article and its table-of-contents entry: the lowercased title, intro, references, category labels, section headings, paragraphs and bullets, joined by spaces (examples and notes are not included), with `&`, `"`, `<` and `>` escaped.
4. `guides/Tax_Strategies.docx`, and the `guides/Tax_Strategies.html` copy.

After changing strategy text, rebuild the chat embeddings (`POST /api/admin/build-embeddings`, in batches of up to 40 via `offset`/`limit`). Embeddings are keyed by title, so a renamed strategy isn't found by the chatbot's search until they are rebuilt.

## Keeping it current

Tax figures change every year, and legislation can change whole entries. At least once a year (after the IRS publishes inflation adjustments, usually in October/November) and after any major tax act:

- Check every dollar threshold and limit in the guide (SALT cap, § 179 and bonus depreciation, § 461(l), QSBS, retirement-plan limits, gift and estate exemptions, QBI thresholds).
- Re-check entries that depend on pending litigation or expiring provisions (for example, the self-employment tax limited-partner cases, and the Qualified Opportunity Zone transition).
- Update the review note in the masthead of `guides/index.html`.

The guide is for learning and internal reference only; it is not tax advice.
