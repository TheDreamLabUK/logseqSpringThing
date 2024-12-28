export class ErrorDisplay {
    private container: HTMLElement;
    private messageElement: HTMLElement;
    private dismissButton: HTMLButtonElement;

    constructor() {
        this.container = document.createElement('div');
        this.container.className = 'error-display hidden';
        
        this.messageElement = document.createElement('div');
        this.messageElement.className = 'error-message';
        
        this.dismissButton = document.createElement('button');
        this.dismissButton.textContent = '×';
        this.dismissButton.className = 'dismiss-button';
        this.dismissButton.addEventListener('click', () => this.hide());
        
        this.container.appendChild(this.messageElement);
        this.container.appendChild(this.dismissButton);
        
        document.body.appendChild(this.container);
    }

    public show(message: string, timeout?: number): void {
        this.messageElement.textContent = message;
        this.container.classList.remove('hidden');
        
        if (timeout) {
            setTimeout(() => this.hide(), timeout);
        }
    }

    public hide(): void {
        this.container.classList.add('hidden');
    }

    public dispose(): void {
        this.dismissButton.removeEventListener('click', () => this.hide());
        this.container.remove();
    }
} 