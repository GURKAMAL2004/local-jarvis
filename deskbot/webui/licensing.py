"""Premium unlock codes.

deskbot is a local, single-machine, open-source tool with no backend server
— there is nowhere to automatically check "did this person actually pay?"
That part is inherently manual: you check the wallet address yourself and
decide a payment came in. What *is* solved properly here is code forgery.

A license code is an Ed25519 signature (made with a PRIVATE key only you
hold) over the buyer's email address. This module ships only the PUBLIC
key, so even though this file is sitting in a public GitHub repo, nobody
can mint a valid code without your private key — the same trust model real
commercial license keys use. Do not confuse "the code can't be forged" with
"the app verifies the payment" — it doesn't and can't; that check is on you.

To issue a code after you've manually confirmed a payment:
    python -m deskbot.webui.generate_license <email> <base64-private-key>
"""

from __future__ import annotations

import base64

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

# Safe to ship: this is the PUBLIC half of the keypair. The private half is
# never stored in this repo — it's generated once and kept by the developer.
PUBLIC_KEY_B64 = "JB+/vt5obHyBD9PAzkPYtCPW9CbCE/nDmw+RxYIVpk4="


def _normalize_email(email: str) -> bytes:
    return email.strip().lower().encode("utf-8")


def verify_license_code(email: str, code: str) -> bool:
    """True only if `code` is a signature, made by the private key matching
    PUBLIC_KEY_B64, over this exact email address."""
    if not email or not code:
        return False
    try:
        signature = base64.b64decode(code.strip(), validate=True)
        public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(PUBLIC_KEY_B64))
        public_key.verify(signature, _normalize_email(email))
        return True
    except (InvalidSignature, ValueError):
        return False
