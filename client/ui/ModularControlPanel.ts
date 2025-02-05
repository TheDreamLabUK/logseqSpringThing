import { SettingsStore } from '../state/SettingsStore';
import { getAllSettingPaths, formatSettingName } from '../types/settings/utils';
import { ValidationErrorDisplay } from '../components/settings/ValidationErrorDisplay';
import { createLogger } from '../core/logger';
import { platformManager } from '../platform/platformManager';
import { nostrAuth, NostrUser } from '../services/NostrAuthService';

const logger = createLogger('ModularControlPanel');

interface SectionConfig {
    id: string;
    title: string;
    isDetached: boolean;
    position?: { x: number; y: number };
    size?: { width: number; height: number };
    isCollapsed: boolean;
    isAdvanced: boolean;
}

export class ModularControlPanel {
    private static instance: ModularControlPanel | null = null;
    private readonly container: HTMLDivElement;
    private readonly settingsStore: SettingsStore;
    private readonly validationDisplay: ValidationErrorDisplay;
    private readonly unsubscribers: Array<() => void> = [];
    private readonly sections: Map<string, SectionConfig> = new Map();
    private updateTimeout: number | null = null;

    private constructor(parentElement: HTMLElement) {
        this.settingsStore = SettingsStore.getInstance();
        
        // Create main container
        this.container = document.createElement('div');
        this.container.id = 'modular-control-panel';
        parentElement.appendChild(this.container);

        // Initialize validation error display
        this.validationDisplay = new ValidationErrorDisplay(this.container);

        // Check platform and settings before showing panel
        if (platformManager.isQuest()) {
            this.hide();
        }

        this.initializePanel();
        this.initializeDragAndDrop();
        this.initializeNostrAuth();
    }

    private async initializePanel(): Promise<void> {
        try {
            await this.settingsStore.initialize();
            
            const settings = this.settingsStore.get('') as any;
            const paths = getAllSettingPaths(settings);
            
            // Group settings by category
            const groupedSettings = this.groupSettingsByCategory(paths);
            
            // Create sections for each category
            for (const [category, categoryPaths] of Object.entries(groupedSettings)) {
                const sectionConfig: SectionConfig = {
                    id: category,
                    title: formatSettingName(category),
                    isDetached: false,
                    isCollapsed: false,
                    isAdvanced: this.isAdvancedCategory(category)
                };
                
                this.sections.set(category, sectionConfig);
                const section = await this.createSection(sectionConfig, categoryPaths);
                this.container.appendChild(section);
            }
            
            logger.info('Modular control panel initialized');
        } catch (error) {
            logger.error('Failed to initialize modular control panel:', error);
        }
    }

    private async initializeNostrAuth(): Promise<void> {
        // Create auth section
        const authSection = document.createElement('div');
        authSection.className = 'settings-section auth-section';
        
        const header = document.createElement('div');
        header.className = 'section-header';
        header.innerHTML = '<h4>Authentication</h4>';
        authSection.appendChild(header);

        const content = document.createElement('div');
        content.className = 'section-content';

        // Create login button
        const loginBtn = document.createElement('button');
        loginBtn.className = 'nostr-login-btn';
        loginBtn.onclick = async () => {
            try {
                // Check if window.nostr is available
                if (!window.nostr) {
                    throw new Error('No Nostr provider found. Please install a Nostr extension.');
                }

                // Request public key from extension
                const pubkey = await window.nostr.getPublicKey();
                if (!pubkey) {
                    throw new Error('Failed to get public key from Nostr extension');
                }

                // Attempt login with pubkey
                const result = await nostrAuth.login();
                if (result.authenticated) {
                    this.updateAuthUI(result.user);
                } else {
                    throw new Error(result.error || 'Authentication failed');
                }
            } catch (error) {
                logger.error('Nostr login failed:', error);
                // Show error in UI
                const errorMsg = document.createElement('div');
                errorMsg.className = 'auth-error';
                errorMsg.textContent = error instanceof Error ? error.message : 'Login failed';
                content.appendChild(errorMsg);
                setTimeout(() => errorMsg.remove(), 3000);
            }
        };

        // Create status display
        const statusDisplay = document.createElement('div');
        statusDisplay.className = 'auth-status';
        
        content.appendChild(loginBtn);
        content.appendChild(statusDisplay);
        authSection.appendChild(content);

        // Add auth section to container
        this.container.insertBefore(authSection, this.container.firstChild);

        // Subscribe to auth state changes
        this.unsubscribers.push(
            nostrAuth.onAuthStateChanged(({ user }) => {
                this.updateAuthUI(user);
            })
        );

        // Initialize auth state
        await nostrAuth.initialize();
        this.updateAuthUI(nostrAuth.getCurrentUser());
    }

    private updateAuthUI(user: NostrUser | null | undefined): void {
        const loginBtn = this.container.querySelector('.nostr-login-btn') as HTMLButtonElement;
        const statusDisplay = this.container.querySelector('.auth-status') as HTMLDivElement;

        if (user) {
            loginBtn.textContent = 'Logout';
            loginBtn.onclick = () => nostrAuth.logout();
            statusDisplay.innerHTML = `
                <div class="user-info">
                    <div class="pubkey">${user.pubkey.substring(0, 8)}...</div>
                    <div class="role">${user.isPowerUser ? 'Power User' : 'Basic User'}</div>
                </div>
            `;
        } else {
            loginBtn.textContent = 'Login with Nostr';
            loginBtn.onclick = () => nostrAuth.login();
            statusDisplay.innerHTML = '<div class="not-authenticated">Not authenticated</div>';
        }
    }

    private initializeDragAndDrop(): void {
        this.container.addEventListener('mousedown', (e: MouseEvent) => {
            const target = e.target as HTMLElement;
            const section = target.closest('.settings-section') as HTMLElement;
            
            if (!section || !target.classList.contains('section-header')) return;
            
            const sectionId = section.dataset.sectionId;
            if (!sectionId) return;

            const sectionConfig = this.sections.get(sectionId);
            if (!sectionConfig) return;

            if (sectionConfig.isDetached) {
                this.startDragging(section, e);
            }
        });
    }

    private startDragging(element: HTMLElement, e: MouseEvent): void {
        const rect = element.getBoundingClientRect();
        const offsetX = e.clientX - rect.left;
        const offsetY = e.clientY - rect.top;

        const moveHandler = (e: MouseEvent) => {
            const x = e.clientX - offsetX;
            const y = e.clientY - offsetY;
            
            element.style.left = `${x}px`;
            element.style.top = `${y}px`;
            
            const sectionId = element.dataset.sectionId;
            if (sectionId) {
                const config = this.sections.get(sectionId);
                if (config) {
                    config.position = { x, y };
                }
            }
        };

        const upHandler = () => {
            document.removeEventListener('mousemove', moveHandler);
            document.removeEventListener('mouseup', upHandler);
        };

        document.addEventListener('mousemove', moveHandler);
        document.addEventListener('mouseup', upHandler);
    }

    private isAdvancedCategory(category: string): boolean {
        const advancedCategories = ['physics', 'rendering', 'debug', 'network'];
        return advancedCategories.includes(category.toLowerCase());
    }

    private async createSection(config: SectionConfig, paths: string[]): Promise<HTMLElement> {
        const section = document.createElement('div');
        section.className = `settings-section ${config.isAdvanced ? 'advanced' : 'basic'}`;
        section.dataset.sectionId = config.id;
        
        if (config.isDetached) {
            section.classList.add('detached');
            if (config.position) {
                section.style.left = `${config.position.x}px`;
                section.style.top = `${config.position.y}px`;
            }
        }

        const header = this.createSectionHeader(config);
        section.appendChild(header);

        const content = document.createElement('div');
        content.className = 'section-content';
        
        // Group paths by subcategory
        const subcategories = this.groupBySubcategory(paths);
        
        for (const [subcategory, subPaths] of Object.entries(subcategories)) {
            const subsection = await this.createSubsection(subcategory, subPaths);
            content.appendChild(subsection);
        }
        
        section.appendChild(content);
        return section;
    }

    private createSectionHeader(config: SectionConfig): HTMLElement {
        const header = document.createElement('div');
        header.className = 'section-header';
        
        const title = document.createElement('h4');
        title.textContent = config.title;
        header.appendChild(title);

        const controls = document.createElement('div');
        controls.className = 'section-controls';

        // Detach button
        const detachBtn = document.createElement('button');
        detachBtn.className = 'section-control detach';
        detachBtn.innerHTML = config.isDetached ? '📌' : '📎';
        detachBtn.title = config.isDetached ? 'Dock section' : 'Detach section';
        detachBtn.onclick = (e) => {
            e.stopPropagation();
            this.toggleDetached(config.id);
        };
        controls.appendChild(detachBtn);

        // Collapse button
        const collapseBtn = document.createElement('button');
        collapseBtn.className = 'section-control collapse';
        collapseBtn.innerHTML = '▼';
        collapseBtn.onclick = (e) => {
            e.stopPropagation();
            this.toggleCollapsed(config.id);
        };
        controls.appendChild(collapseBtn);

        header.appendChild(controls);
        return header;
    }

    private groupSettingsByCategory(paths: string[]): Record<string, string[]> {
        const groups: Record<string, string[]> = {};
        
        paths.forEach(path => {
            const [category] = path.split('.');
            if (!groups[category]) {
                groups[category] = [];
            }
            groups[category].push(path);
        });
        
        return groups;
    }

    private groupBySubcategory(paths: string[]): Record<string, string[]> {
        const groups: Record<string, string[]> = {};
        
        paths.forEach(path => {
            const parts = path.split('.');
            if (parts.length > 2) {
                const subcategory = parts[1];
                if (!groups[subcategory]) {
                    groups[subcategory] = [];
                }
                groups[subcategory].push(path);
            } else {
                if (!groups['general']) {
                    groups['general'] = [];
                }
                groups['general'].push(path);
            }
        });
        
        return groups;
    }

    private async createSubsection(subcategory: string, paths: string[]): Promise<HTMLElement> {
        const subsection = document.createElement('div');
        subsection.className = 'settings-subsection';
        
        const header = document.createElement('h3');
        header.textContent = formatSettingName(subcategory);
        header.className = 'settings-subsection-header';
        subsection.appendChild(header);
        
        for (const path of paths) {
            const control = await this.createSettingControl(path);
            subsection.appendChild(control);
        }
        
        return subsection;
    }

    private async createSettingControl(path: string): Promise<HTMLElement> {
        // Implementation placeholder - using path parameter
        const control = document.createElement('div');
        control.dataset.settingPath = path;
        return control;
    }

    private toggleDetached(sectionId: string): void {
        const config = this.sections.get(sectionId);
        if (!config) return;

        config.isDetached = !config.isDetached;
        const section = this.container.querySelector(`[data-section-id="${sectionId}"]`);
        if (section) {
            section.classList.toggle('detached');
            if (config.isDetached) {
                const rect = section.getBoundingClientRect();
                config.position = { x: rect.left, y: rect.top };
            } else {
                (section as HTMLElement).removeAttribute('style');
            }
        }
    }

    private toggleCollapsed(sectionId: string): void {
        const config = this.sections.get(sectionId);
        if (!config) return;

        config.isCollapsed = !config.isCollapsed;
        const section = this.container.querySelector(`[data-section-id="${sectionId}"]`);
        if (section) {
            section.classList.toggle('collapsed');
        }
    }

    public static getInstance(): ModularControlPanel {
        if (!ModularControlPanel.instance) {
            ModularControlPanel.instance = new ModularControlPanel(document.body);
        }
        return ModularControlPanel.instance;
    }

    public show(): void {
        this.container.classList.add('visible');
    }

    public hide(): void {
        this.container.classList.remove('visible');
    }

    public dispose(): void {
        this.unsubscribers.forEach(unsubscribe => unsubscribe());
        this.validationDisplay.dispose();
        if (this.updateTimeout !== null) {
            window.clearTimeout(this.updateTimeout);
        }
        this.container.remove();
        ModularControlPanel.instance = null;
    }
}