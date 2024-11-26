interface ErrorDetails {
  message: string;
  stack?: string;
  component?: string;
  context?: string;
  timestamp: number;
  userAgent: string;
  url: string;
  additional?: Record<string, any>;
}

class ErrorTrackingService {
  private static instance: ErrorTrackingService;
  private errors: ErrorDetails[] = [];
  private maxErrors = 100;
  private isDebugMode: boolean;

  private constructor() {
    this.isDebugMode = window.location.search.includes('debug') || process.env.NODE_ENV === 'development';
    this.setupGlobalHandlers();
  }

  public static getInstance(): ErrorTrackingService {
    if (!ErrorTrackingService.instance) {
      ErrorTrackingService.instance = new ErrorTrackingService();
    }
    return ErrorTrackingService.instance;
  }

  private setupGlobalHandlers() {
    // Handle Vue errors
    window.__VUE_PROD_DEVTOOLS__ = true;
    window.__VUE_PROD_ERROR_HANDLER__ = (err: Error, vm: any, info: string) => {
      this.trackError(err, {
        context: 'Vue Error',
        component: vm?.$options?.name || 'Unknown Component',
        additional: { info }
      });
    };

    // Handle promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.trackError(event.reason, {
        context: 'Unhandled Promise Rejection'
      });
    });

    // Handle runtime errors
    window.addEventListener('error', (event) => {
      this.trackError(event.error || new Error(event.message), {
        context: 'Runtime Error',
        additional: {
          filename: event.filename,
          lineno: event.lineno,
          colno: event.colno
        }
      });
    });
  }

  private formatError(error: unknown): Error {
    if (error instanceof Error) {
      return error;
    }
    
    if (typeof error === 'string') {
      return new Error(error);
    }
    
    try {
      const errorMessage = error instanceof Object 
        ? JSON.stringify(error)
        : String(error);
      return new Error(errorMessage);
    } catch {
      return new Error('Unknown error');
    }
  }

  public trackError(
    error: unknown,
    options: {
      context?: string;
      component?: string;
      additional?: Record<string, any>;
    } = {}
  ): ErrorDetails {
    const formattedError = this.formatError(error);
    
    const errorDetails: ErrorDetails = {
      message: formattedError.message,
      stack: formattedError.stack,
      component: options.component,
      context: options.context,
      timestamp: Date.now(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      additional: options.additional
    };

    // Add to local storage
    this.errors.push(errorDetails);
    if (this.errors.length > this.maxErrors) {
      this.errors.shift();
    }

    // Log to console in debug mode
    if (this.isDebugMode) {
      console.error('Error tracked:', {
        ...errorDetails,
        context: options.context || 'Error Tracking Service'
      });
    }

    // Store in localStorage for persistence
    try {
      localStorage.setItem('error_log', JSON.stringify(this.errors));
    } catch (e) {
      console.warn('Failed to store error log in localStorage');
    }

    // Emit custom event for debug panel
    window.dispatchEvent(new CustomEvent('error-tracked', {
      detail: errorDetails
    }));

    return errorDetails;
  }

  public getErrors(): ErrorDetails[] {
    return this.errors;
  }

  public clearErrors(): void {
    this.errors = [];
    try {
      localStorage.removeItem('error_log');
    } catch (e) {
      console.warn('Failed to clear error log from localStorage');
    }
  }

  public downloadErrorLog(): void {
    const errorLog = JSON.stringify(this.errors, null, 2);
    const blob = new Blob([errorLog], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `error-log-${new Date().toISOString()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  public getErrorSummary(): {
    total: number;
    types: Record<string, number>;
    components: Record<string, number>;
  } {
    return {
      total: this.errors.length,
      types: this.errors.reduce((acc, err) => {
        const type = err.context || 'Unknown';
        acc[type] = (acc[type] || 0) + 1;
        return acc;
      }, {} as Record<string, number>),
      components: this.errors.reduce((acc, err) => {
        if (err.component) {
          acc[err.component] = (acc[err.component] || 0) + 1;
        }
        return acc;
      }, {} as Record<string, number>)
    };
  }
}

// Export singleton instance
export const errorTracking = ErrorTrackingService.getInstance();
