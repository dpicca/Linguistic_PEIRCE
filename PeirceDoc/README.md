# PEIRCE + LMM Documentation

This folder contains the MkDocs configuration and Markdown sources for the PEIRCE + LMM documentation site.

## Build prerequisites

Install documentation dependencies (from project root):

```bash
pip install -r requirements.txt
```

This will install MkDocs and mkdocstrings.

## Build the site

Run the following command from the project root:

```bash
mkdocs build -f PeirceDoc/mkdocs.yml
```

The static site will be generated in the `site/` folder inside `PeirceDoc/` by default.

## Serve locally

To preview the docs locally with live reload:

```bash
mkdocs serve -f PeirceDoc/mkdocs.yml
```

Then open the URL printed by MkDocs in your browser.

## Content

- The homepage (index.md) is generated from the repository README content.
- The API Reference (api.md) is automatically populated from Python docstrings via mkdocstrings.

If you add new modules or functions, ensure they have comprehensive Google-style docstrings so they render well in the API reference.
