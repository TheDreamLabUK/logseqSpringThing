# WebSocket Direct Connection Test Plan

## Current Setup
The current architecture uses nginx to handle WebSocket connections with these features:
```nginx
location /ws {
    proxy_pass http://backend;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host $host;
    
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
    proxy_connect_timeout 30s;
}
```

## Hypothesis
Nginx might be unnecessary because:
1. Cloudflared already handles:
   - WebSocket protocol upgrades
   - Connection timeouts
   - Keep-alive settings
2. Docker internal networking provides:
   - Direct container-to-container communication
   - Network isolation
   - Name resolution

## Test Steps

### 1. Configuration Changes
1. Remove nginx container entirely
2. Update cloudflared config to point directly to webxr:3000:
```yaml
ingress:
  - hostname: www.visionflow.info
    service: http://logseq-xr-webxr:3000  # Changed from 4000
    originRequest:
      noTLSVerify: true
      connectTimeout: 30s
      tcpKeepAlive: 30s
      keepAliveTimeout: 2m
      keepAliveConnections: 100
      httpHostHeader: www.visionflow.info
      idleTimeout: 3600s
      streamTimeout: 3600s
```

3. Update docker-compose.yml:
   - Remove nginx service
   - Update webxr port mapping if needed
   - Keep docker_ragflow network unchanged

### 2. Test Cases

#### A. Basic Connectivity
1. Access https://www.visionflow.info
2. Verify initial page load
3. Check browser console for connection errors

#### B. WebSocket Specific
1. Monitor WebSocket connection status
2. Verify WebSocket upgrade headers in browser network tab
3. Test real-time updates and data streaming
4. Check if multiple WebSocket connections work correctly

#### C. Connection Management
1. Test connection stability over time
2. Verify timeout handling
3. Check reconnection behavior
4. Monitor memory usage without nginx buffer

#### D. Multiple Clients
1. Connect multiple browsers simultaneously
2. Verify each maintains separate WebSocket connection
3. Test concurrent data streaming
4. Check for any connection conflicts

### 3. Monitoring Points

1. Client Side:
   - Browser WebSocket connection status
   - Network tab timing and headers
   - Console errors or warnings
   - Performance metrics

2. Server Side:
   - webxr container logs
   - cloudflared tunnel logs
   - Connection counts
   - Memory usage

### 4. Success Criteria

1. Must Have:
   - Successful WebSocket connections
   - Stable real-time updates
   - No connection drops
   - Proper error handling

2. Should Have:
   - Similar or better performance than with nginx
   - Clean connection termination
   - Proper client IP handling
   - Multiple client support

### 5. Rollback Plan

If issues occur:
1. Keep original nginx.conf
2. Document specific failures
3. Analyze if issues are related to:
   - Protocol handling
   - Header management
   - Connection multiplexing
   - Other nginx-specific features

## Potential Concerns

1. WebSocket Protocol:
   - Does the Rust backend expect specific headers that nginx provides?
   - Are there any custom headers needed for WebSocket upgrade?

2. Connection Management:
   - How does the backend handle multiple WebSocket connections?
   - Is there any connection pooling that nginx provides?

3. Security:
   - Does nginx provide any critical security headers?
   - Are there any rate limiting features we need?

4. Routing:
   - How does the backend distinguish between WebSocket and HTTP traffic?
   - Are there any path-specific handlers that nginx manages?

## Next Steps

1. Implementation:
   - Create backup of current configuration
   - Make changes in staging environment first
   - Test thoroughly before production

2. Monitoring:
   - Set up enhanced logging
   - Monitor connection patterns
   - Track performance metrics

3. Documentation:
   - Update network documentation
   - Document any issues found
   - Create troubleshooting guide

This test plan will help verify if nginx is truly surplus to requirements while ensuring we understand any potential impacts of its removal.
2. Connection Management:
   - How does the backend handle multiple WebSocket connections?
   - Is there any connection pooling that nginx provides?

3. Security:
   - Does nginx provide any critical security headers?
   - Are there any rate limiting features we need?

4. Routing:
   - How does the backend distinguish between WebSocket and HTTP traffic?
   - Are there any path-specific handlers that nginx manages?

## Next Steps

1. Implementation:
   - Create backup of current configuration
   - Make changes in staging environment first
   - Test thoroughly before production

2. Monitoring:
   - Set up enhanced logging
   - Monitor connection patterns
   - Track performance metrics

3. Documentation:
   - Update network documentation
   - Document any issues found
   - Create troubleshooting guide

This test plan will help verify if nginx is truly surplus to requirements while ensuring we understand any potential impacts of its removal.
