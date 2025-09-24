#!/usr/bin/env bash
set -euo pipefail

echo "==> Starting entrypoint"

# Optional: wait for DB if vars provided
if [[ -n "${HOST:-}" && -n "${DB_PORT:-}" ]]; then
  echo "==> Waiting for DB ${HOST}:${DB_PORT} ..."
  for i in {1..40}; do
    if nc -z "${HOST}" "${DB_PORT}" >/dev/null 2>&1; then
      echo "==> DB is reachable."
      break
    fi
    echo "   ...still waiting ($i/40)"
    sleep 2
  done
fi

# Verify WSGI import BEFORE gunicorn forks (fail fast, readable error)
python - <<'PY'
import importlib, os, sys
mod = os.getenv("DJANGO_SETTINGS_MODULE", "cammouflagebackend.settings")
print(f"DJANGO_SETTINGS_MODULE={mod}")
m = importlib.import_module("cammouflagebackend.wsgi")  # <— change if single “m”
print("WSGI import OK:", m)
PY

# Try migrations (don’t block app if they fail)
echo "==> Running migrations..."
python manage.py migrate --noinput || echo "!! Migrations failed — app will still start so you can see logs."

# Optional: collect static for DEBUG=False (skip if not needed)
if [[ "${COLLECTSTATIC:-0}" == "1" ]]; then
  echo "==> collectstatic ..."
  python manage.py collectstatic --noinput || echo "!! collectstatic failed"
fi

echo "==> Handing off to: $*"
exec "$@"
