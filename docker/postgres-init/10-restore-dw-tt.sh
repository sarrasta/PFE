#!/usr/bin/env bash
# Restores the DW_TT data warehouse from the original project's pg_dump
# custom-format backup (`ml pfe/datasql/todhia.backup`, bind-mounted read-only
# into this container by docker-compose.yml — never copied, so the dataset
# only exists once on disk).
#
# Runs automatically ONLY on first container start, when $PGDATA is empty —
# that's how docker-entrypoint-initdb.d works. `docker compose down -v` +
# `up` re-triggers it; a plain restart of an existing volume does not.
#
# Deliberately restores WITHOUT --create: the dump's embedded
# "CREATE DATABASE ... LOCALE = 'French_France.1252'" statement references a
# Windows code page that doesn't exist on this Linux/glibc image. POSTGRES_DB
# below has already created an empty "DW_TT" database with the container's
# own (portable) default locale by the time this script runs, so we restore
# schema + data into it directly and skip that one problematic statement.
set -euo pipefail

BACKUP_FILE="/warehouse-dump/todhia.backup"
TARGET_DB="${POSTGRES_DB:-DW_TT}"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "[restore-dw-tt] WARNING: $BACKUP_FILE not found — skipping restore." \
         "Check the volume mount in docker-compose.yml." >&2
    exit 0
fi

echo "[restore-dw-tt] Restoring $TARGET_DB from $BACKUP_FILE ..."
pg_restore \
    --username "$POSTGRES_USER" \
    --dbname "$TARGET_DB" \
    --no-owner \
    --no-privileges \
    --clean \
    --if-exists \
    --exit-on-error \
    "$BACKUP_FILE"

echo "[restore-dw-tt] Restore complete."
