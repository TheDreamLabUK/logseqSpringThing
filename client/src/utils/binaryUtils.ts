/**
 * Utility functions for binary data handling
 * 
 * Note: Binary decompression is now handled by the graph worker.
 * This file is kept for potential future binary utilities.
 */
import { createLogger } from './logger';

const logger = createLogger('BinaryUtils');

// This file previously contained decompression utilities that have been
// moved to the graph worker for better performance. If you need binary
// utilities in the future, add them here.

export {};