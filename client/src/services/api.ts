import { createLogger, createErrorMetadata } from '../utils/logger';

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
   * Make a GET request to the API
   */
  public async get<T>(endpoint: string, headers: Record<string, string> = {}): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...headers
        }
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      logger.error(`GET request to ${endpoint} failed:`, createErrorMetadata(error));
      throw error;
    }
  }

  /**
   * Make a POST request to the API
   */
  public async post<T>(endpoint: string, data: any, headers: Record<string, string> = {}): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...headers
        },
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      logger.error(`POST request to ${endpoint} failed:`, createErrorMetadata(error));
      throw error;
    }
  }

  /**
   * Make a DELETE request to the API
   */
  public async delete<T>(endpoint: string, headers: Record<string, string> = {}): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          ...headers
        }
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      logger.error(`DELETE request to ${endpoint} failed:`, createErrorMetadata(error));
      throw error;
    }
  }
}

export const apiService = ApiService.getInstance();
