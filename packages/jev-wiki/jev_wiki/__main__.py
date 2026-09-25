"""Allow ``python -m jev_wiki`` without importing the parent Cognee package."""

from .cli import main

raise SystemExit(main())
