import { createLogger, createErrorMetadata } from '../utils/logger';
import { debugState } from '../utils/debugState';
import { RagflowChatRequestPayload, RagflowChatResponsePayload } from '../types/ragflowTypes';

const logger = createLogger('ApiService');

/**
 * API Service for making requests to the backend
 */
class ApiService {
  private static instance: ApiService;
  private baseUrl: string;

  private constructor() {
    this.baseUrl = '/api';
  }

  public static getInstance(): ApiService {
    if (!ApiService.instance) {
      ApiService.instance = new ApiService();
    }
    return ApiService.instance;
  }

  /**
   * Set the base URL for API requests
   * @param url The new base URL
   */
  public setBaseUrl(url: string): void {
    this.baseUrl = url;
    logger.info(`API base URL set to: ${url}`);
  }

  /**
   * Get the current base URL
   */
  public getBaseUrl(): string {
    return this.baseUrl;
  }

  /**
   * Make a GET request to the API
   * @param endpoint The API endpoint
   * @param headers Optional request headers
   * @returns The response data
   */
  public async get<T>(endpoint: string, headers: Record<string, string> = {}): Promise<T> {
    try {
      const url = `${this.baseUrl}${endpoint}`;

      if (debugState.isEnabled()) {
        logger.debug(`Making GET request to ${url}`);
      }

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...headers
        }
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (debugState.isEnabled()) {
        logger.debug(`GET request to ${endpoint} succeeded`);
      }

      return data;
    } catch (error) {
      logger.error(`GET request to ${endpoint} failed:`, createErrorMetadata(error));
      throw error;
    }
  }

  /**
   * Make a POST request to the API
   * @param endpoint The API endpoint
   * @param data The request body data
   * @param headers Optional request headers
   * @returns The response data
   */
  public async post<T>(endpoint: string, data: any, headers: Record<string, string> = {}): Promise<T> {
    try {
      const url = `${this.baseUrl}${endpoint}`;

      if (debugState.isEnabled()) {
        logger.debug(`Making POST request to ${url}`);
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...headers
        },
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}: ${response.statusText}`);
      }

      const responseData = await response.json();

      if (debugState.isEnabled()) {
        logger.debug(`POST request to ${endpoint} succeeded`);
      }

      return responseData;
    } catch (error) {
      logger.error(`POST request to ${endpoint} failed:`, createErrorMetadata(error));
      throw error;
    }
  }

  /**
   * Make a PUT request to the API
   * @param endpoint The API endpoint
   * @param data The request body data
   * @param headers Optional request headers
   * @returns The response data
   */
  public async put<T>(endpoint: string, data: any, headers: Record<string, string> = {}): Promise<T> {
    try {
      const url = `${this.baseUrl}${endpoint}`;

      if (debugState.isEnabled()) {
        logger.debug(`Making PUT request to ${url}`);
      }

      const response = await fetch(url, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          ...headers
        },
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}: ${response.statusText}`);
      }

      const responseData = await response.json();

      if (debugState.isEnabled()) {
        logger.debug(`PUT request to ${endpoint} succeeded`);
      }

      return responseData;
    } catch (error) {
      logger.error(`PUT request to ${endpoint} failed:`, createErrorMetadata(error));
      throw error;
    }
  }

  /**
   * Make a DELETE request to the API
   * @param endpoint The API endpoint
   * @param headers Optional request headers
   * @returns The response data
   */
  public async delete<T>(endpoint: string, headers: Record<string, string> = {}): Promise<T> {
    try {
      const url = `${this.baseUrl}${endpoint}`;

      if (debugState.isEnabled()) {
        logger.debug(`Making DELETE request to ${url}`);
      }

      const response = await fetch(url, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          ...headers
        }
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      if (debugState.isEnabled()) {
        logger.debug(`DELETE request to ${endpoint} succeeded`);
      }

      return data;
    } catch (error) {
      logger.error(`DELETE request to ${endpoint} failed:`, createErrorMetadata(error));
      throw error;
    }
  }

  public async sendRagflowChatMessage(
    payload: RagflowChatRequestPayload,
    headers: Record<string, string> = {} // For auth
  ): Promise<RagflowChatResponsePayload> {
    try {
      const url = `${this.baseUrl}/ragflow/chat`; // Path defined in Rust backend, /api is prepended by baseUrl
      if (debugState.isEnabled()) {
        logger.debug(`Making POST request to ${url} for RAGFlow chat`);
      }
      // Assume headers (like auth) will be passed in or handled globally by apiService
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...headers, // Include auth headers
        },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
            const errorData = await response.json().catch(() => ({ message: response.statusText, error: response.statusText }));
            throw new Error(`RAGFlow chat API request failed with status ${response.status}: ${errorData.error || errorData.message}`);
          }
          const responseData = await response.json();
          if (debugState.isEnabled()) {
            logger.debug(`POST request to ${url} (RAGFlow chat) succeeded`);
          }
          return responseData as RagflowChatResponsePayload;
        } catch (error) {
          logger.error(`POST request to /ragflow/chat failed:`, createErrorMetadata(error));
          throw error;
        }
      }
}

export const apiService = ApiService.getInstance();
