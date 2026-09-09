"""Registered configuration migrations for adapter-owned options."""

import importlib

# Composition defaults; extensions can register another migration without changing TaskConfig.
MIGRATIONS = ["threadweave.coding_config:migrate_legacy"]


def migrate_options(value, fields):
    if not isinstance(value, dict):
        return value
    migrated = dict(value)
    for reference in MIGRATIONS:
        module, name = reference.split(":")
        migrated = getattr(importlib.import_module(module), name)(migrated, fields)
    return migrated


def migrate_session(value):
    if not isinstance(value, dict):
        return value
    from .coding_config import migrate_session as migrate

    return migrate(value)
