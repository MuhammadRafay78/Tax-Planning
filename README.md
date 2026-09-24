# Tax-Planning

A reference guide of U.S. tax-planning strategies, published as a static site on Cloudflare Workers.

## Layout

| Path | What it is |
|---|---|
| `guides/index.html` | The guide: all strategies, table of contents, category filter and theme switch in one self-contained page. |
| `guides/Tax_Strategies.docx` | Word version of the guide. |
| `guides/404.html` | Not-found page. |
| `guides/_redirects` | Sends the old `/Tax_Strategies.html` URL to `/`. |
| `wrangler.jsonc` | Cloudflare Workers static-assets config; serves the `guides/` directory. |

## Adding or editing a strategy

Each strategy is an `<article class="strategy">` in `guides/index.html`. When adding one:

1. Give it the next `id`/`data-index` and add a matching entry to the table of contents (`#toc`).
2. Set `data-cats` on both the article and the TOC entry, and update the counts in the category `<select>` and the sidebar strategy count.
3. Cite code sections, regulations and cases in the "Key Code References" list.

## Keeping it current

Tax figures change every year, and legislation can change whole entries. At least once a year (after the IRS publishes inflation adjustments, usually in October/November) and after any major tax act:

- Check every dollar threshold and limit in the guide (SALT cap, § 179 and bonus depreciation, § 461(l), QSBS, retirement-plan limits, gift and estate exemptions, QBI thresholds).
- Re-check entries that depend on pending litigation or expiring provisions (for example, the self-employment tax limited-partner cases, and the Qualified Opportunity Zone transition).
- Update the "as of" date in the masthead of `guides/index.html`.
- Regenerate or update `guides/Tax_Strategies.docx` so it matches the site.

The guide is for learning and internal reference only; it is not tax advice.
