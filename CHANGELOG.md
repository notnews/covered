# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Adopted [py-canon](https://github.com/gojiplus/py-canon) fleet standards: CI
  now runs via the reusable `py-canon` workflow (ruff, pyright, pydoclint,
  pytest, zizmor, dependency review); switched from mypy to pyright for type
  checking; dropped the `<3.13` upper bound on `requires-python` (no
  dependency actually requires it); versioning is now derived from git tags
  via `hatch-vcs` instead of a hardcoded string.
