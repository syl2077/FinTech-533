# Breakout Project Workflow

1. Start IB Gateway or TWS in paper mode so `shinybroker` can request data from `127.0.0.1:7497`.
2. From the project root, run:

   ```bash
   ../.venv-1/bin/python breakout_project.py --force-download
   ```

   To force a specific asset instead of the default symbol, use:

   ```bash
   ../.venv-1/bin/python breakout_project.py --asset NVDA --force-download
   ```

   To screen several assets and let the script choose the best one, use:

   ```bash
   ../.venv-1/bin/python breakout_project.py --symbols MU AAPL NVDA GLD --force-download
   ```

3. Re-render the site:

   ```bash
   env XDG_CACHE_HOME=/tmp/quarto-cache DENO_DIR=/tmp/deno-dir HOME=/tmp/quarto-home quarto render
   ```

4. Commit the updated `project_outputs/` source files plus the regenerated `docs/` site output.

The analysis code lives at the project root because Quarto clears `docs/` before each render. The published GitHub Pages artifacts still end up in `docs/`, and `docs/pull_data.py` remains there as the original ShinyBroker reference structure.
