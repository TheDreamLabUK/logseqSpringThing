/**
 * WebSocket Diagnostics Tool
 * 
 * This tool provides comprehensive diagnostics for WebSocket connections,
 * including connection status, message handling, binary protocol validation,
 * and network analysis.
 * 
 * Usage:
 * 1. Import this file in your application or load it via the browser console
 * 2. Call `runDiagnostics()` to start the diagnostics process
 * 3. Check the console for detailed logs and diagnostics information
 * 
 * Features:
 * - Connection status monitoring
 * - Binary protocol validation
 * - Network latency measurement
 * - Message size analysis
 * - Reconnection testing
 */

import { buildWsUrl } from './core/api';
import pako from 'pako';
import { debugState } from './core/debugState';
import { logger } from './core/logger';

// Configuration
const CONFIG = {
  // Timeouts and intervals (in milliseconds)
  connectionTimeout: 10000,
  pingInterval: 2000,
  reconnectDelay: 3000,
  testDuration: 30000,
  
  // Test parameters
  connectionAttempts: 3,
  binaryValidationSamples: 5,
  
  // Expected binary protocol values
  expectedBytesPerNode: 28,
  expectedHeaderSize: 8,
  
  // Logging
  verbose: true,
  logTimestamps: true
};

// Diagnostic state
const state = {
  connectionAttempts: 0,
  messagesReceived: 0,
  binaryMessagesReceived: 0,
  textMessagesReceived: 0,
  reconnections: 0,
  errors: 0,
  binarySizes: [] as number[],
  latencies: [] as number[],
  pingTimestamps: new Map<string, number>(),
  testStartTime: 0,
  socket: null as WebSocket | null,
  testRunning: false
};

// Utility functions
const utils = {
  timestamp(): string {
    return CONFIG.logTimestamps 
      ? `[${new Date().toISOString()}] `
      : '';
  },
  
  log(message: string, type: 'info' | 'error' | 'warning' | 'success' = 'info'): void {
    const prefix = utils.timestamp();
    
    switch (type) {
      case 'error':
        console.error(`${prefix}❌ ${message}`);
        break;
      case 'warning':
        console.warn(`${prefix}⚠️ ${message}`);
        break;
      case 'success':
        console.log(`${prefix}✅ ${message}`);
        break;
      default:
        console.log(`${prefix}ℹ️ ${message}`);
    }
  },
  
  formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },
  
  calculateAverage(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  },
  
  generateRandomId(): string {
    return Math.random().toString(36).substring(2, 15);
  },
  
  isCompressed(data: ArrayBuffer): boolean {
    // Check for zlib header (78 01, 78 9C, or 78 DA)
    const header = new Uint8Array(data, 0, 2);
    return header[0] === 0x78 && (header[1] === 0x01 || header[1] === 0x9C || header[1] === 0xDA);
  }
};

// Network diagnostics
const network = {
  async testApiConnectivity(): Promise<boolean> {
    try {
      utils.log('Testing API connectivity...');
      const response = await fetch('/api/user-settings', { method: 'HEAD' });
      if (response.ok) {
        utils.log('API connectivity test passed', 'success');
        return true;
      } else {
        utils.log(`API connectivity test failed: ${response.status} ${response.statusText}`, 'error');
        return false;
      }
    } catch (error) {
      utils.log(`API connectivity test failed: ${error}`, 'error');
      return false;
    }
  },
  
  async checkDnsResolution(): Promise<boolean> {
    try {
      utils.log('Checking DNS resolution...');
      const hostname = window.location.hostname;
      const response = await fetch(`https://${hostname}/favicon.ico`, { method: 'HEAD' });
      if (response.ok) {
        utils.log(`DNS resolution successful: ${response.status}`, 'success');
      } else {
        utils.log(`DNS resolution returned status: ${response.status}`, 'warning');
      }
      return response.ok;
    } catch (error) {
      utils.log(`DNS resolution failed: ${error}`, 'error');
      return false;
    }
  }
};

// Binary protocol validation
const binaryProtocol = {
  validateMessageSize(data: ArrayBuffer): boolean {
    const size = data.byteLength;
    state.binarySizes.push(size);
    
    // Check if the message is compressed
    if (utils.isCompressed(data)) {
      utils.log(`Received compressed binary message: ${utils.formatBytes(size)}`);
      try {
        // Decompress the message
        const compressedData = new Uint8Array(data);
        const decompressedData = pako.inflate(compressedData);
        return this.validateDecompressedMessage(decompressedData.buffer);
      } catch (error) {
        utils.log(`Failed to decompress message: ${error}`, 'error');
        return false;
      }
    } else {
      return this.validateDecompressedMessage(data);
    }
  },
  
  validateDecompressedMessage(data: ArrayBuffer): boolean {
    const size = data.byteLength;
    const headerSize = CONFIG.expectedHeaderSize;
    const bytesPerNode = CONFIG.expectedBytesPerNode;
    
    // Check if the message size (minus header) is divisible by the expected bytes per node
    const dataSize = size - headerSize;
    const remainder = dataSize % bytesPerNode;
    
    if (remainder !== 0) {
      utils.log(`Binary message size validation failed: message size (${size} bytes) minus header (${headerSize} bytes) = ${dataSize} bytes, which is not divisible by ${bytesPerNode} bytes per node. Remainder: ${remainder} bytes`, 'error');
      return false;
    }
    
    const nodeCount = dataSize / bytesPerNode;
    utils.log(`Binary message contains data for ${nodeCount} nodes (${utils.formatBytes(size)})`, 'info');
    
    // Read the header to get the actual node count
    const view = new DataView(data);
    const messageType = view.getUint32(0, true);
    const reportedNodeCount = view.getUint32(4, true);
    
    utils.log(`Message type: ${messageType}, reported node count: ${reportedNodeCount}`, 'info');
    
    if (nodeCount !== reportedNodeCount) {
      utils.log(`Node count mismatch: header reports ${reportedNodeCount} nodes, but message contains data for ${nodeCount} nodes`, 'error');
      return false;
    }
    
    utils.log(`Binary message validation passed: ${reportedNodeCount} nodes, ${utils.formatBytes(size)}`, 'success');
    return true;
  },
  
  analyzeMessageFrequency(): void {
    if (state.binaryMessagesReceived === 0) {
      utils.log('No binary messages received during the test period', 'warning');
      return;
    }
    
    const avgSize = utils.calculateAverage(state.binarySizes);
    const messagesPerSecond = state.binaryMessagesReceived / (CONFIG.testDuration / 1000);
    
    utils.log(`Binary message frequency: ${messagesPerSecond.toFixed(2)} messages/second`);
    utils.log(`Average binary message size: ${utils.formatBytes(avgSize)}`);
  }
};

// WebSocket connection handling
const wsConnection = {
  connect(): Promise<WebSocket> {
    return new Promise((resolve, reject) => {
      state.connectionAttempts++;
      
      const url = buildWsUrl();
      utils.log(`Connecting to WebSocket: ${url} (Attempt ${state.connectionAttempts}/${CONFIG.connectionAttempts})`);
      
      const socket = new WebSocket(url);
      state.socket = socket;
      
      // Set up connection timeout
      const timeoutId = setTimeout(() => {
        if (socket.readyState !== WebSocket.OPEN) {
          utils.log('WebSocket connection timeout', 'error');
          socket.close();
          reject(new Error('Connection timeout'));
        }
      }, CONFIG.connectionTimeout);
      
      socket.onopen = () => {
        clearTimeout(timeoutId);
        utils.log('WebSocket connection established', 'success');
        
        // Send initial message to request data
        this.sendInitialRequest(socket);
        
        // Start ping test
        this.startPingTest(socket);
        
        resolve(socket);
      };
      
      socket.onerror = (error) => {
        clearTimeout(timeoutId);
        state.errors++;
        utils.log(`WebSocket error: ${error}`, 'error');
        reject(error);
      };
      
      socket.onclose = (event) => {
        clearTimeout(timeoutId);
        utils.log(`WebSocket connection closed: Code ${event.code}, Reason: ${event.reason || 'No reason provided'}`);
        
        if (state.testRunning && state.connectionAttempts < CONFIG.connectionAttempts) {
          utils.log(`Attempting to reconnect in ${CONFIG.reconnectDelay}ms...`);
          state.reconnections++;
          
          setTimeout(() => {
            this.connect().catch(error => {
              utils.log(`Reconnection failed: ${error}`, 'error');
            });
          }, CONFIG.reconnectDelay);
        }
      };
      
      socket.onmessage = (event) => {
        this.handleMessage(event);
      };
    });
  },
  
  sendInitialRequest(socket: WebSocket): void {
    if (socket.readyState === WebSocket.OPEN) {
      const initialRequest = JSON.stringify({ type: 'requestInitialData' });
      socket.send(initialRequest);
      utils.log('Sent initial data request');
    }
  },
  
  startPingTest(socket: WebSocket): void {
    const pingInterval = setInterval(() => {
      if (socket.readyState !== WebSocket.OPEN) {
        clearInterval(pingInterval);
        return;
      }
      
      const pingId = utils.generateRandomId();
      const pingMessage = JSON.stringify({ type: 'ping', id: pingId });
      
      state.pingTimestamps.set(pingId, Date.now());
      socket.send(pingMessage);
      
      if (CONFIG.verbose) {
        utils.log(`Sent ping: ${pingId}`);
      }
    }, CONFIG.pingInterval);
  },
  
  handleMessage(event: MessageEvent): void {
    state.messagesReceived++;
    
    if (typeof event.data === 'string') {
      state.textMessagesReceived++;
      this.handleTextMessage(event.data);
    } else if (event.data instanceof ArrayBuffer) {
      state.binaryMessagesReceived++;
      this.handleBinaryMessage(event.data);
    } else if (event.data instanceof Blob) {
      // Convert Blob to ArrayBuffer
      const reader = new FileReader();
      reader.onload = () => {
        if (reader.result instanceof ArrayBuffer) {
          state.binaryMessagesReceived++;
          this.handleBinaryMessage(reader.result);
        }
      };
      reader.readAsArrayBuffer(event.data);
    }
  },
  
  handleTextMessage(data: string): void {
    try {
      const message = JSON.parse(data);
      
      if (CONFIG.verbose) {
        utils.log(`Received text message: ${JSON.stringify(message)}`);
      }
      
      // Handle pong messages for latency calculation
      if (message.type === 'pong' && message.id) {
        const pingTime = state.pingTimestamps.get(message.id);
        if (pingTime) {
          const latency = Date.now() - pingTime;
          state.latencies.push(latency);
          state.pingTimestamps.delete(message.id);
          
          if (CONFIG.verbose) {
            utils.log(`Received pong: ${message.id}, latency: ${latency}ms`);
          }
        }
      }
    } catch (error) {
      utils.log(`Failed to parse text message: ${error}`, 'error');
    }
  },
  
  handleBinaryMessage(data: ArrayBuffer): void {
    if (CONFIG.verbose) {
      utils.log(`Received binary message: ${utils.formatBytes(data.byteLength)}`);
    }
    
    // Validate binary message format
    binaryProtocol.validateMessageSize(data);
  }
};

// Main diagnostics functions
export async function runDiagnostics(): Promise<void> {
  utils.log('Starting WebSocket diagnostics...');
  state.testStartTime = Date.now();
  state.testRunning = true;
  
  try {
    // Check network connectivity
    const apiConnectivity = await network.testApiConnectivity();
    if (!apiConnectivity) {
      utils.log('API connectivity test failed, but continuing with WebSocket tests', 'warning');
    }
    
    // Connect to WebSocket
    const socket = await wsConnection.connect();
    
    // Run the test for the configured duration
    setTimeout(() => {
      state.testRunning = false;
      if (socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
      
      // Generate diagnostics report
      generateReport();
    }, CONFIG.testDuration);
  } catch (error) {
    utils.log(`Diagnostics failed: ${error}`, 'error');
    state.testRunning = false;
    generateReport();
  }
}

function generateReport(): void {
  utils.log('--- WebSocket Diagnostics Report ---');
  utils.log(`Test duration: ${(Date.now() - state.testStartTime) / 1000} seconds`);
  utils.log(`Connection attempts: ${state.connectionAttempts}`);
  utils.log(`Reconnections: ${state.reconnections}`);
  utils.log(`Errors: ${state.errors}`);
  utils.log(`Total messages received: ${state.messagesReceived}`);
  utils.log(`Text messages received: ${state.textMessagesReceived}`);
  utils.log(`Binary messages received: ${state.binaryMessagesReceived}`);
  
  if (state.latencies.length > 0) {
    const avgLatency = utils.calculateAverage(state.latencies);
    const minLatency = Math.min(...state.latencies);
    const maxLatency = Math.max(...state.latencies);
    
    utils.log(`Latency - Avg: ${avgLatency.toFixed(2)}ms, Min: ${minLatency}ms, Max: ${maxLatency}ms`);
  } else {
    utils.log('No latency measurements available', 'warning');
  }
  
  binaryProtocol.analyzeMessageFrequency();
  
  // Provide recommendations based on diagnostics
  provideRecommendations();
}

function provideRecommendations(): void {
  utils.log('--- Recommendations ---');
  
  if (state.errors > 0) {
    utils.log('⚠️ Connection errors detected. Check network stability and server availability.');
  }
  
  if (state.reconnections > 0) {
    utils.log('⚠️ Multiple reconnections detected. This may indicate network instability or server issues.');
  }
  
  if (state.binaryMessagesReceived === 0) {
    utils.log('⚠️ No binary messages received. Check if the server is sending updates or if the initial request was processed.');
  }
  
  if (state.latencies.length > 0) {
    const avgLatency = utils.calculateAverage(state.latencies);
    if (avgLatency > 200) {
      utils.log('⚠️ High average latency detected. This may affect real-time performance.');
    }
  }
  
  utils.log('✅ Diagnostics complete. Use this information to troubleshoot WebSocket issues.');
}

// Export functions for manual invocation
export const WebSocketDiagnostics = {
  runDiagnostics,
  testApiConnectivity: network.testApiConnectivity,
  checkDnsResolution: network.checkDnsResolution,
  CONFIG
};

// Auto-run diagnostics if in development mode
if (process.env.NODE_ENV === 'development') {
  if (debugState.isWebsocketDebugEnabled()) {
    logger.debug('WebSocket diagnostics tool loaded. Call WebSocketDiagnostics.runDiagnostics() to start diagnostics.');
  }
} 