# Security Documentation

## Overview

This section covers all security-related aspects of LogseqSpringThing, including authentication, authorization, API security, and best practices for deployment.

## Topics

### [Authentication](./authentication.md)
Comprehensive guide to the Nostr-based authentication system, including:
- NIP-07 browser extension integration
- Session management
- API key handling
- Role-based access control (RBAC)

### Binary Protocol Security
The [Binary Protocol Documentation](../api/binary-protocol.md) includes security considerations for:
- Input validation
- Memory safety
- Rate limiting
- Data integrity

### WebSocket Security
The [WebSocket API Documentation](../api/websocket.md) covers:
- Authentication requirements for WebSocket connections
- Message validation
- Connection security
- Rate limiting

## Quick Reference

### Environment Variables

```bash
# Authentication
AUTH_TOKEN_EXPIRY=3600  # Session token expiry in seconds
POWER_USER_PUBKEYS=pubkey1,pubkey2,pubkey3  # Comma-separated power users

# API Keys (for power users)
PERPLEXITY_API_KEY=your-key
OPENAI_API_KEY=your-key
RAGFLOW_API_KEY=your-key
```

### Security Headers

All authenticated API requests must include:
```
X-Nostr-Pubkey: <user-public-key>
Authorization: Bearer <session-token>
```

### CORS Configuration

Configure allowed origins in `protected_settings.json`:
```json
{
  "security": {
    "allowed_origins": [
      "http://localhost:3000",
      "https://your-domain.com"
    ]
  }
}
```

## Security Checklist

### Development
- [ ] Use environment variables for sensitive data
- [ ] Enable debug logging for auth services
- [ ] Test with multiple NIP-07 extensions
- [ ] Verify session expiry handling

### Production
- [ ] Use HTTPS for all connections
- [ ] Configure proper CORS origins
- [ ] Set appropriate session timeouts
- [ ] Enable audit logging
- [ ] Configure rate limiting
- [ ] Use strong API keys
- [ ] Regular security updates
- [ ] Monitor for suspicious activity

## Common Security Patterns

### 1. Authenticated API Request

```typescript
const response = await fetch('/api/protected-endpoint', {
  headers: {
    'X-Nostr-Pubkey': user.pubkey,
    'Authorization': `Bearer ${sessionToken}`,
    'Content-Type': 'application/json'
  }
});
```

### 2. WebSocket Authentication

```typescript
const ws = new WebSocket(`wss://your-domain/wss?token=${sessionToken}`);
```

### 3. Feature Access Check

```typescript
const hasAccess = await nostrAuth.checkFeatureAccess('premium-feature');
if (!hasAccess) {
  throw new Error('Feature requires premium access');
}
```

## Security Best Practices

### 1. Authentication
- Always verify Nostr event signatures server-side
- Use secure session token generation (UUID v4)
- Implement proper session expiry
- Clear sessions on logout

### 2. Authorization
- Check feature access for protected operations
- Validate user roles before API key operations
- Use environment variables for power user configuration
- Implement least privilege principle

### 3. Data Protection
- Sanitize all user inputs
- Use parameterized queries for database operations
- Implement rate limiting on sensitive endpoints
- Log security events for auditing

### 4. Network Security
- Use TLS/SSL for all connections
- Implement proper CORS policies
- Validate WebSocket message formats
- Set appropriate timeout values

### 5. Client Security
- Store tokens securely (localStorage/sessionStorage)
- Clear sensitive data on logout
- Validate server responses
- Implement client-side rate limiting

## Threat Model

### Potential Threats

1. **Session Hijacking**
   - Mitigated by: HTTPS, secure token generation, session expiry

2. **Unauthorized Access**
   - Mitigated by: Nostr signature verification, role-based access control

3. **API Key Exposure**
   - Mitigated by: Environment variables, power user restrictions

4. **DoS Attacks**
   - Mitigated by: Rate limiting, connection limits, message size limits

5. **Data Tampering**
   - Mitigated by: Binary protocol validation, bounds checking

## Incident Response

### Security Issue Reporting

Report security vulnerabilities to: security@logseqspringthing.com

### Response Process

1. **Detection**: Monitor logs for suspicious activity
2. **Containment**: Revoke compromised sessions/tokens
3. **Investigation**: Analyze audit logs
4. **Remediation**: Apply security patches
5. **Communication**: Notify affected users

## Additional Resources

- [Nostr Protocol Specification](https://github.com/nostr-protocol/nips)
- [NIP-07: Web Browser Signer](https://github.com/nostr-protocol/nips/blob/master/07.md)
- [NIP-42: Authentication](https://github.com/nostr-protocol/nips/blob/master/42.md)
- [OWASP Security Guidelines](https://owasp.org/)