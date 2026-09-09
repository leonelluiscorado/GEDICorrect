# Changelog

All notable changes to GEDICorrect are documented in this file.

## 1.0.0 - Unreleased

### Added

- Installable Python package with `gedicorrect` and `gedicorrect-ui` commands.
- Shared validated configuration and execution API.
- Local Streamlit browser interface with background processing and job logs.
- Linux container and Docker Compose workflow for Windows, macOS, and Linux.
- Windows launcher with native data-folder dialogs, automatic Docker startup, and reusable configuration.
- Automatic LAS and GEDI dataset discovery in the container UI.
- Persistent UI job monitoring across browser refreshes and new browser sessions.
- Terminal-aware live logs that render `tqdm` progress updates cleanly in the browser.
- Reactive random-candidate controls that disable incompatible grid settings.
- Container-aware CPU detection and conservative numerical thread limits.
- Environment diagnostics through `gedicorrect check`.
- Automated tests for configuration, CLI, runner, and system-resource handling.

### Changed

- Moved the correction implementation into the `gedicorrect` package namespace.
- Forwarded `time_window` correctly from user interfaces to the correction engine.
- Improved external command failure detection and error reporting.
- Simplified the Conda environment and removed its machine-specific prefix.
- Updated installation, UI, Docker, CLI, Python API, paper, and citation documentation.

### Removed

- Undeclared runtime dependency on `memory_profiler`.
