"""Best-effort event redaction; artifact bytes remain exact and require private storage."""

import os
import re

SENSITIVE = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token|credential)", re.I
)


def redact(value, credential_names=()):
    secrets = {
        v
        for k, v in os.environ.items()
        if len(v) >= 8 and (k in credential_names or SENSITIVE.search(k))
    }

    def clean(item):
        if isinstance(item, dict):
            return {
                k: "[REDACTED]" if SENSITIVE.search(k) and not k.endswith("_env") else clean(v)
                for k, v in item.items()
            }
        if isinstance(item, list):
            return [clean(v) for v in item]
        if isinstance(item, str):
            for secret in secrets:
                item = item.replace(secret, "[REDACTED]")
        return item

    return clean(value)
