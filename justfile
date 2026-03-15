set shell := ["powershell", "-NoProfile", "-Command"]

rb *args:
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts/rb.ps1 {{args}}

wsl *args:
  wsl -d Ubuntu-24.04 -e bash scripts/run_wsl.sh {{args}}
