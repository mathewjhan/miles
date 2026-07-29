"""Error types for the session module.

Hierarchy
---------
SessionError (base)
├── SessionNotFoundError       → 404  session does not exist
├── MessageValidationError     → 400  messages structure/content invalid
├── SessionContextLimitExceededError → 400  session token budget exhausted
├── TokenizationError          → 500  TITO tokenizer / prefix mismatch
└── UpstreamResponseError      → 502  SGLang response invalid or unexpected
"""


class SessionError(Exception):
    """Base class for all session-related errors."""

    status_code: int = 500


class SessionNotFoundError(SessionError):
    """Raised when the requested session ID does not exist."""

    status_code: int = 404


class MessageValidationError(SessionError):
    """Raised when request messages fail structural validation.

    Examples: user message after assistant, messages not append-only,
    rollback failed (no assistant checkpoint in matched prefix).
    """

    status_code: int = 400


class SessionContextLimitExceededError(SessionError):
    """Raised when a session's next prompt exceeds the training max_seq_len.

    Mirrors SGLang's own context-length 400 (clients see the same class of
    terminal error), but fires at max_seq_len — typically far below the
    model's context window — so runaway trajectories stop before sampling
    tokens the trainer would discard anyway.
    """

    status_code: int = 400


class TokenizationError(SessionError):
    """Raised when TITO tokenization invariants are violated.

    Examples: pretokenized prefix mismatch between stored and new token IDs.
    """

    status_code: int = 500


class UpstreamResponseError(SessionError):
    """Raised when the upstream SGLang response is invalid or unexpected.

    Examples: missing meta_info, assistant content is None,
    output_token_logprobs length mismatch.
    """

    status_code: int = 502
