"""
Sphinx configuration for FlagOS-Compressor documentation.
"""

from __future__ import annotations

# -- Project information -----------------------------------------------------

project = "FlagOS-Compressor Documentation"
copyright = "2026, FlagOS Community"
author = "FlagOS Community"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinxext.opengraph",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "_includes",
]
master_doc = "index"

autosectionlabel_prefix_document = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "strikethrough",
    "substitution",
    "tasklist",
    "attrs_inline",
    "attrs_block",
]
myst_heading_anchors = 2

# -- Internationalization ----------------------------------------------------

language = "en"
locale_dirs = ["locale/"]
gettext_compact = False

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/custom.js"]
html_title = "FlagOS-Compressor Documentation"

html_theme_options = {
    "home_page_in_toc": True,
    "navbar_persistent": [],
    "repository_url": "https://github.com/flagos-ai/FlagOS-Compressor",
    "use_download_button": False,
    "use_edit_page_button": True,
    "use_repository_button": True,
}

htmlhelp_basename = "FlagOSCompressordoc"

# -- OpenGraph ---------------------------------------------------------------

ogp_site_name = "FlagOS-Compressor Documentation"
ogp_use_first_image = True
ogp_enable_meta_description = True
ogp_description_length = 300

# -- Intersphinx -------------------------------------------------------------

intersphinx_cache_limit = 14
intersphinx_timeout = 3
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}

# -- Linkcheck ---------------------------------------------------------------

linkcheck_retries = 2
linkcheck_timeout = 5
linkcheck_workers = 10
linkcheck_ignore = [
    r"http://127\.0\.0\.1",
    r"http://localhost",
    r"https://github\.com.+?#L\d+",
]

# -- Extensions configuration -------------------------------------------------

extlinks = {
    "issue": ("https://github.com/flagos-ai/FlagOS-Compressor/issues/%s", "#%s"),
}

suppress_warnings = ["epub.unknown_project_files"]
