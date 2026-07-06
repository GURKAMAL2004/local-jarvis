"""Developer-only tool for minting premium license codes.

NOT part of the app end users run — you run this yourself, locally, after
manually confirming a crypto payment landed in your wallet. Needs your
PRIVATE key (the one paired with licensing.PUBLIC_KEY_B64), which you should
keep secret and never commit to the repo.

Usage:
    python -m deskbot.webui.generate_license <email> <base64-private-key>
"""

from __future__ import annotations

import base64
import sys

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def generate_code(email: str, private_key_b64: str) -> str:
    private_key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
    signature = private_key.sign(email.strip().lower().encode("utf-8"))
    return base64.b64encode(signature).decode("ascii")


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python -m deskbot.webui.generate_license <email> <base64-private-key>")
        return 1
    print(generate_code(sys.argv[1], sys.argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
