interface ErrorInfo {
  message: string;
  context?: string;
  component?: string;
  stack?: string;
  timestamp: number;
  additional?: any;
}

class ErrorTrackingService {
  private errors: ErrorInfo[] = [];
  private maxErrors = 100;

  private constructor() {
    this.setupGlobalHandlers();
  }

  private static instance: ErrorTrackingService;

  public static getInstance(): ErrorTrackingService {
    if (!ErrorTrackingService.instance) {
      ErrorTrackingService.instance = new ErrorTrackingService();
    }
    return ErrorTrackingService.instance;
  }

  private setupGlobalHandlers() {
    window.onerror = (message, source, lineno, colno, error) => {
      this.trackError(error || new Error(String(message)), {
        context: 'Global Error Handler',
        additional: { source, lineno, colno }
      });
    };

    window.onunhandledrejection = (event) => {
      this.trackError(event.reason, {
        context: 'Unhandled Promise Rejection'
      });
    };
  }

  public trackError(error: Error | unknown, info?: {
    context?: string;
    component?: string;
    additional?: any;
  }) {
    const errorInfo: ErrorInfo = {
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
      context: info?.context,
      component: info?.component,
      additional: info?.additional,
      timestamp: Date.now()
    };

    this.errors.unshift(errorInfo);

    // Trim old errors
    if (this.errors.length > this.maxErrors) {
      this.errors = this.errors.slice(0, this.maxErrors);
    }

    // Emit event for any listeners
    window.dispatchEvent(new CustomEvent('error-tracked', {
      detail: errorInfo
    }));
  }

  public getErrors(): ErrorInfo[] {
    return [...this.errors];
  }

  public clearErrors(): void {
    this.errors = [];
  }
}

export const errorTracking = ErrorTrackingService.getInstance();
