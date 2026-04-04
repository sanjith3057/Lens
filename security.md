# Security Checklist & Documentation

## Core Security Pillars
1. **Input Sanitization**: Layer 1 (Parser) handles prompt injection and PII redaction.
2. **Key Management**: `SecureConfig` ensures API keys are loaded from environment variables and never hardcoded.
3. **Rate Limiting**: `RateLimiter` prevents API abuse.
4. **Logic Isolation**: Multi-layered architecture ensures that vision model calls are isolated from data parsing.

## Implemented Mitigations
- Prompt injection detection in `parser.py`.
- PII redaction for SSN and Credit Card patterns.
- Rate limiting for Groq and Ollama APIs.
- Secure environment configuration via `.env`.
