/**
 * WebSocket Connection Test Script
 * 
 * This script can be run in the browser console to diagnose WebSocket connection issues.
 * It tests the WebSocket connection to the server and logs detailed information about
 * the connection process, including any errors that occur.
 * 
 * Usage:
 * 1. Open the browser console (F12 or Ctrl+Shift+I)
 * 2. Copy and paste this entire script into the console
 * 3. Press Enter to run the test
 * 4. Check the console for detailed logs about the WebSocket connection
 */

// Configuration
const TEST_TIMEOUT_MS = 10000; // 10 seconds
const PING_INTERVAL_MS = 2000; // 2 seconds
const CONNECTION_ATTEMPTS = 3;  // Number of connection attempts

// Utility functions
function log(message: string, data?: any) {
  const timestamp = new Date().toISOString();
  if (data) {
    console.log(`[${timestamp}] ${message}`, data);
  } else {
    console.log(`[${timestamp}] ${message}`);
  }
}

function error(message: string, err?: any) {
  const timestamp = new Date().toISOString();
  if (err) {
    console.error(`[${timestamp}] ERROR: ${message}`, err);
  } else {
    console.error(`[${timestamp}] ERROR: ${message}`);
  }
}

function warn(message: string, data?: any) {
  const timestamp = new Date().toISOString();
  if (data) {
    console.warn(`[${timestamp}] WARNING: ${message}`, data);
  } else {
    console.warn(`[${timestamp}] WARNING: ${message}`);
  }
}

// Build WebSocket URL using the same logic as the application
function buildWsUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.hostname;
  // Check if we're in production (any visionflow.info domain)
  const isProduction = host.endsWith('visionflow.info');
  const port = isProduction ? '' : ':4000';
  const base = `${protocol}//${host}${port}`;
  const wsPath = '/wss';
  return `${base}${wsPath}`;
}

// Test network connectivity to the API
async function testApiConnectivity() {
  log('Testing API connectivity...');
  try {
    const response = await fetch('/api/user-settings', { 
      method: 'HEAD',
      cache: 'no-cache'
    });
    log('API connectivity test result:', {
      status: response.status,
      ok: response.ok,
      statusText: response.statusText
    });
    return response.ok;
  } catch (err) {
    error('API connectivity test failed', err);
    return false;
  }
}

// Test WebSocket connection
function testWebSocketConnection(url: string): Promise<boolean> {
  return new Promise((resolve) => {
    log(`Testing WebSocket connection to ${url}...`);
    
    let pingInterval: number | null = null;
    let connectionTimeout: number | null = null;
    
    const ws = new WebSocket(url);
    
    // Set binary type to arraybuffer (same as the application)
    ws.binaryType = 'arraybuffer';
    
    // Set connection timeout
    connectionTimeout = window.setTimeout(() => {
      error('WebSocket connection timed out');
      if (pingInterval) clearInterval(pingInterval);
      ws.close();
      resolve(false);
    }, TEST_TIMEOUT_MS);
    
    ws.onopen = () => {
      log('WebSocket connection established successfully');
      
      // Clear connection timeout
      if (connectionTimeout) {
        clearTimeout(connectionTimeout);
        connectionTimeout = null;
      }
      
      // Send a ping message
      const pingMessage = JSON.stringify({
        type: 'ping',
        timestamp: Date.now()
      });
      ws.send(pingMessage);
      log('Sent ping message');
      
      // Set up ping interval
      pingInterval = window.setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          const pingMessage = JSON.stringify({
            type: 'ping',
            timestamp: Date.now()
          });
          ws.send(pingMessage);
          log('Sent ping message');
        }
      }, PING_INTERVAL_MS);
      
      // Send requestInitialData message (same as the application)
      const requestMessage = JSON.stringify({
        type: 'requestInitialData'
      });
      ws.send(requestMessage);
      log('Sent requestInitialData message');
      
      // Resolve after 5 seconds of successful connection
      setTimeout(() => {
        if (pingInterval) clearInterval(pingInterval);
        ws.close();
        resolve(true);
      }, 5000);
    };
    
    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        try {
          const message = JSON.parse(event.data);
          log('Received message:', message);
        } catch (err) {
          warn('Received non-JSON string message:', event.data);
        }
      } else if (event.data instanceof ArrayBuffer) {
        const buffer = event.data;
        log('Received binary message', {
          byteLength: buffer.byteLength,
          isMultipleOf28: buffer.byteLength % 28 === 0
        });
        
        // Try to decompress using pako if available
        if (window.pako) {
          try {
            const decompressed = window.pako.inflate(new Uint8Array(buffer));
            log('Decompressed binary message', {
              originalSize: buffer.byteLength,
              decompressedSize: decompressed.byteLength,
              isMultipleOf28: decompressed.byteLength % 28 === 0
            });
          } catch (err) {
            // Not compressed or invalid data
            log('Binary message is not compressed or invalid');
          }
        }
      }
    };
    
    ws.onerror = (event) => {
      error('WebSocket error occurred', event);
    };
    
    ws.onclose = (event) => {
      if (connectionTimeout) {
        clearTimeout(connectionTimeout);
        connectionTimeout = null;
      }
      
      if (pingInterval) {
        clearInterval(pingInterval);
        pingInterval = null;
      }
      
      log('WebSocket connection closed', {
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean
      });
      
      // If not already resolved
      if (connectionTimeout !== null) {
        resolve(false);
      }
    };
  });
}

// Run all tests
async function runTests() {
  log('Starting WebSocket connection tests');
  log('Browser information:', {
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    vendor: navigator.vendor
  });
  
  log('Page URL:', window.location.href);
  
  // Test API connectivity first
  const apiConnectivity = await testApiConnectivity();
  if (!apiConnectivity) {
    error('API connectivity test failed. WebSocket connection is unlikely to succeed.');
  }
  
  // Build WebSocket URL
  const wsUrl = buildWsUrl();
  log('WebSocket URL:', wsUrl);
  
  // Parse URL to check components
  try {
    const parsedUrl = new URL(wsUrl);
    log('WebSocket URL components:', {
      protocol: parsedUrl.protocol,
      host: parsedUrl.host,
      hostname: parsedUrl.hostname,
      port: parsedUrl.port,
      pathname: parsedUrl.pathname,
      search: parsedUrl.search
    });
    
    // Check if using secure WebSocket
    if (parsedUrl.protocol !== 'wss:' && window.location.protocol === 'https:') {
      warn('Using insecure WebSocket (ws://) with HTTPS site - browsers may block this');
    }
  } catch (err) {
    error('Failed to parse WebSocket URL', err);
  }
  
  // Test WebSocket connection multiple times
  let successCount = 0;
  for (let i = 0; i < CONNECTION_ATTEMPTS; i++) {
    log(`Connection attempt ${i + 1} of ${CONNECTION_ATTEMPTS}`);
    const success = await testWebSocketConnection(wsUrl);
    if (success) {
      successCount++;
    }
    
    // Wait between attempts
    if (i < CONNECTION_ATTEMPTS - 1) {
      log('Waiting 2 seconds before next attempt...');
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }
  
  // Report results
  log(`WebSocket test complete. ${successCount} of ${CONNECTION_ATTEMPTS} connection attempts succeeded.`);
  
  if (successCount === 0) {
    error('All WebSocket connection attempts failed. Please check your network configuration and server status.');
  } else if (successCount < CONNECTION_ATTEMPTS) {
    warn(`${CONNECTION_ATTEMPTS - successCount} of ${CONNECTION_ATTEMPTS} connection attempts failed. Connection may be unstable.`);
  } else {
    log('All WebSocket connection attempts succeeded. Connection is stable.');
  }
}

// Start the tests
runTests().catch(err => {
  error('Unhandled error during tests', err);
});

// Export for use in console
(window as any).testWebSocket = {
  runTests,
  testApiConnectivity,
  testWebSocketConnection,
  buildWsUrl
};

// Log completion message
log('WebSocket test script loaded. Tests are running automatically.');
log('You can also run tests manually using: testWebSocket.runTests()'); 