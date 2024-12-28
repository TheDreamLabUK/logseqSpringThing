export class LoadingOverlay {
    private overlay: HTMLElement;
    private startTime: number = 0;
    private minDisplayTime: number = 500; // Minimum time to show loading overlay

    constructor() {
        this.overlay = document.createElement('div');
        this.overlay.className = 'loading-overlay hidden';
        this.overlay.innerHTML = `
            <div class="spinner"></div>
            <div class="loading-text">Loading...</div>
        `;
        document.body.appendChild(this.overlay);
    }

    public show(): void {
        this.startTime = performance.now();
        this.overlay.classList.remove('hidden');
    }

    public async hide(): Promise<void> {
        const elapsedTime = performance.now() - this.startTime;
        const remainingTime = Math.max(0, this.minDisplayTime - elapsedTime);
        
        if (remainingTime > 0) {
            await new Promise(resolve => setTimeout(resolve, remainingTime));
        }
        
        this.overlay.classList.add('hidden');
    }

    public dispose(): void {
        this.overlay.remove();
    }
} 