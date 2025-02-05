import { SettingsStore } from '../state/SettingsStore';
import { getAllSettingPaths, formatSettingName } from '../types/settings/utils';
import { ValidationErrorDisplay } from '../components/settings/ValidationErrorDisplay';
import { createLogger } from '../core/logger';
import { platformManager } from '../platform/platformManager';
import { nostrAuth, NostrUser } from '../services/NostrAuthService';
import { EventEmitter } from '../utils/eventEmitter';

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

export interface ModularControlPanelEvents {
    'settings:ready': null;
    'settings:updated': { path: string; value: any };
}

export class ModularControlPanel extends EventEmitter<ModularControlPanelEvents> {
    private static instance: ModularControlPanel | null = null;
    private readonly container: HTMLDivElement;
    private readonly toggleButton: HTMLButtonElement;
    private readonly settingsStore: SettingsStore;
    private readonly validationDisplay: ValidationErrorDisplay;
    private readonly unsubscribers: Array<() => void> = [];
    private readonly sections: Map<string, SectionConfig> = new Map();
    private updateTimeout: number | null = null;
    private isInitialized: boolean = false;

    private constructor(parentElement: HTMLElement) {
        super();
        this.settingsStore = SettingsStore.getInstance();
        
        // Create toggle button first
        this.toggleButton = document.createElement('button');
        this.toggleButton.className = 'panel-toggle-btn';
        this.toggleButton.innerHTML = '⚙️';
        this.toggleButton.onclick = () => this.toggle();
        parentElement.appendChild(this.toggleButton);

        // Create main container
        this.container = document.createElement('div');
        this.container.id = 'modular-control-panel';
        parentElement.appendChild(this.container);

        // Initialize validation error display
        this.validationDisplay = new ValidationErrorDisplay(this.container);

        // Check platform and settings before showing panel
        if (platformManager.isQuest()) {
            this.hide();
        } else {
            // Show by default on non-Quest platforms
            this.show();
        }

        this.initializeComponents();
    }

    private async initializeComponents(): Promise<void> {
        try {
            // Initialize settings first
            await this.initializeSettings();
            
            // Then initialize UI components
            await this.initializePanel();
            this.initializeDragAndDrop();
            await this.initializeNostrAuth();
            
            // Mark as initialized and emit ready event
            this.isInitialized = true;
            this.emit('settings:ready', null);
            
            logger.info('ModularControlPanel fully initialized');
        } catch (error) {
            logger.error('Failed to initialize ModularControlPanel:', error);
            throw error;
        }
    }

    private async initializeSettings(): Promise<void> {
        try {
            await this.settingsStore.initialize();
            logger.info('Settings initialized successfully');
        } catch (error) {
            logger.error('Failed to initialize settings:', error);
            throw error;
        }
    }

    private async initializePanel(): Promise<void> {
        try {
            const settings = this.settingsStore.get('') as any;
            const paths = getAllSettingPaths(settings);
            
            // Create main categories container
            const categoriesContainer = document.createElement('div');
            categoriesContainer.className = 'settings-categories';
            
            // Group settings by main category
            const mainCategories = ['visualization', 'system', 'xr'];
            const groupedSettings = this.groupSettingsByCategory(paths);
            
            // Create sections for each main category first
            for (const category of mainCategories) {
                if (groupedSettings[category]) {
                    const sectionConfig: SectionConfig = {
                        id: category,
                        title: formatSettingName(category),
                        isDetached: false,
                        isCollapsed: false,
                        isAdvanced: this.isAdvancedCategory(category)
                    };
                    
                    this.sections.set(category, sectionConfig);
                    const section = await this.createSection(sectionConfig, groupedSettings[category]);
                    categoriesContainer.appendChild(section);
                }
            }
            
            // Add remaining categories
            for (const [category, categoryPaths] of Object.entries(groupedSettings)) {
                if (!mainCategories.includes(category)) {
                    const sectionConfig: SectionConfig = {
                        id: category,
                        title: formatSettingName(category),
                        isDetached: false,
                        isCollapsed: false,
                        isAdvanced: this.isAdvancedCategory(category)
                    };
                    
                    this.sections.set(category, sectionConfig);
                    const section = await this.createSection(sectionConfig, categoryPaths);
                    categoriesContainer.appendChild(section);
                }
            }
            
            this.container.appendChild(categoriesContainer);
            logger.info('Panel UI initialized');
        } catch (error) {
            logger.error('Failed to initialize panel:', error);
            throw error;
        }
    }

    private async initializeNostrAuth(): Promise<void> {
        const authSection = document.createElement('div');
        authSection.className = 'settings-section auth-section';
        
        const header = document.createElement('div');
        header.className = 'section-header';
        header.innerHTML = '<h4>Authentication</h4>';
        authSection.appendChild(header);

        const content = document.createElement('div');
        content.className = 'section-content';

        const loginBtn = document.createElement('button');
        loginBtn.className = 'nostr-login-btn';
        loginBtn.onclick = async () => {
            try {
                if (!window.nostr) {
                    throw new Error('No Nostr provider found. Please install a Nostr extension.');
                }

                const pubkey = await window.nostr.getPublicKey();
                if (!pubkey) {
                    throw new Error('Failed to get public key from Nostr extension');
                }

                const result = await nostrAuth.login();
                if (result.authenticated) {
                    this.updateAuthUI(result.user);
                } else {
                    throw new Error(result.error || 'Authentication failed');
                }
            } catch (error) {
                logger.error('Nostr login failed:', error);
                const errorMsg = document.createElement('div');
                errorMsg.className = 'auth-error';
                errorMsg.textContent = error instanceof Error ? error.message : 'Login failed';
                content.appendChild(errorMsg);
                setTimeout(() => errorMsg.remove(), 3000);
            }
        };

        const statusDisplay = document.createElement('div');
        statusDisplay.className = 'auth-status';
        
        content.appendChild(loginBtn);
        content.appendChild(statusDisplay);
        authSection.appendChild(content);

        this.container.insertBefore(authSection, this.container.firstChild);

        this.unsubscribers.push(
            nostrAuth.onAuthStateChanged(({ user }) => {
                this.updateAuthUI(user);
            })
        );

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
        
        const validPaths = paths.filter(path => {
            const value = this.settingsStore.get(path);
            return value !== undefined && value !== null;
        });
        
        if (validPaths.length > 0) {
            const subcategories = this.groupBySubcategory(validPaths);
            
            const sortedSubcategories = Object.entries(subcategories).sort(([a], [b]) => {
                if (a === 'general') return -1;
                if (b === 'general') return 1;
                return a.localeCompare(b);
            });
            
            for (const [subcategory, subPaths] of sortedSubcategories) {
                if (subPaths.length > 0) {
                    const subsection = await this.createSubsection(subcategory, subPaths);
                    content.appendChild(subsection);
                }
            }
        } else {
            const emptyMessage = document.createElement('div');
            emptyMessage.className = 'empty-section-message';
            emptyMessage.textContent = 'No configurable settings in this section';
            content.appendChild(emptyMessage);
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

        const detachBtn = document.createElement('button');
        detachBtn.className = 'section-control detach';
        detachBtn.innerHTML = config.isDetached ? '📌' : '📎';
        detachBtn.title = config.isDetached ? 'Dock section' : 'Detach section';
        detachBtn.onclick = (e) => {
            e.stopPropagation();
            this.toggleDetached(config.id);
        };
        controls.appendChild(detachBtn);

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
        
        Object.keys(groups).forEach(key => {
            groups[key].sort((a, b) => {
                const aName = a.split('.').pop() || '';
                const bName = b.split('.').pop() || '';
                return aName.localeCompare(bName);
            });
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
            } else if (parts.length === 2) {
                if (!groups['general']) {
                    groups['general'] = [];
                }
                groups['general'].push(path);
            }
        });
        
        Object.keys(groups).forEach(key => {
            groups[key].sort((a, b) => {
                const aName = a.split('.').pop() || '';
                const bName = b.split('.').pop() || '';
                return aName.localeCompare(bName);
            });
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
        const container = document.createElement('div');
        container.className = 'setting-control';
        container.dataset.settingPath = path;

        const label = document.createElement('label');
        label.textContent = formatSettingName(path.split('.').pop() || '');
        container.appendChild(label);

        const currentValue = this.settingsStore.get(path);
        const control = await this.createInputElement(path, currentValue);
        container.appendChild(control);

        return container;
    }

    private async createInputElement(path: string, value: any): Promise<HTMLElement> {
        const type = typeof value;
        let input: HTMLElement;

        switch (type) {
            case 'boolean': {
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.checked = value;
                checkbox.onchange = (e) => {
                    const target = e.target as HTMLInputElement;
                    this.updateSetting(path, target.checked);
                };
                input = checkbox;
                break;
            }

            case 'number': {
                const numberInput = document.createElement('input');
                numberInput.type = 'number';
                numberInput.value = value.toString();
                numberInput.step = '0.01';
                numberInput.onchange = (e) => {
                    const target = e.target as HTMLInputElement;
                    this.updateSetting(path, parseFloat(target.value));
                };
                input = numberInput;
                break;
            }

            case 'string': {
                if (path.toLowerCase().includes('color')) {
                    const colorInput = document.createElement('input');
                    colorInput.type = 'color';
                    colorInput.value = value;
                    colorInput.onchange = (e) => {
                        const target = e.target as HTMLInputElement;
                        this.updateSetting(path, target.value);
                    };
                    input = colorInput;
                } else {
                    const options = this.getOptionsForPath(path);
                    if (options.length > 0) {
                        const select = document.createElement('select');
                        options.forEach(opt => {
                            const option = document.createElement('option');
                            option.value = opt;
                            option.textContent = formatSettingName(opt);
                            option.selected = opt === value;
                            select.appendChild(option);
                        });
                        select.onchange = (e) => {
                            const target = e.target as HTMLSelectElement;
                            this.updateSetting(path, target.value);
                        };
                        input = select;
                    } else {
                        const textInput = document.createElement('input');
                        textInput.type = 'text';
                        textInput.value = value;
                        textInput.onchange = (e) => {
                            const target = e.target as HTMLInputElement;
                            this.updateSetting(path, target.value);
                        };
                        input = textInput;
                    }
                }
                break;
            }

            case 'object': {
                if (Array.isArray(value)) {
                    const textInput = document.createElement('input');
                    textInput.type = 'text';
                    textInput.value = value.join(',');
                    textInput.onchange = (e) => {
                        const target = e.target as HTMLInputElement;
                        this.updateSetting(path, target.value.split(',').map(v => v.trim()));
                    };
                    input = textInput;
                } else {
                    const container = document.createElement('div');
                    container.className = 'nested-object-subsection';
                    
                    const header = document.createElement('div');
                    header.className = 'nested-header';
                    header.textContent = formatSettingName(path.split('.').pop() || '');
                    container.appendChild(header);
                    
                    const content = document.createElement('div');
                    content.className = 'nested-content';
                    
                    for (const [key] of Object.entries(value)) {
                        const nestedPath = `${path}.${key}`;
                        const nestedValue = this.settingsStore.get(nestedPath);
                        if (nestedValue !== undefined) {
                            const control = document.createElement('div');
                            control.className = 'setting-control';
                            
                            const label = document.createElement('label');
                            label.textContent = formatSettingName(key);
                            control.appendChild(label);
                            
                            const nestedInput = await this.createInputElement(nestedPath, nestedValue);
                            control.appendChild(nestedInput);
                            
                            content.appendChild(control);
                        }
                    }
                    
                    container.appendChild(content);
                    input = container;
                }
                break;
            }

            default: {
                const div = document.createElement('div');
                div.className = 'value-display';
                div.textContent = value?.toString() || '';
                input = div;
            }
        }

        return input;
    }

    private getOptionsForPath(path: string): string[] {
        const optionsMap: Record<string, string[]> = {
            'visualization.nodes.quality': ['low', 'medium', 'high'],
            'visualization.edges.quality': ['low', 'medium', 'high'],
            'visualization.rendering.context': ['desktop', 'ar'],
            'system.debug.logLevel': ['error', 'warn', 'info', 'debug', 'trace'],
            'xr.mode': ['immersive-vr', 'immersive-ar'],
            'xr.spaceType': ['local', 'local-floor', 'bounded-floor']
        };

        return optionsMap[path] || [];
    }

    private updateSetting(path: string, value: any): void {
        if (this.updateTimeout !== null) {
            window.clearTimeout(this.updateTimeout);
        }

        this.updateTimeout = window.setTimeout(async () => {
            try {
                await this.settingsStore.set(path, value);
                this.emit('settings:updated', { path, value });
            } catch (error) {
                logger.error(`Failed to update setting ${path}:`, error);
            }
        }, 100);
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

    public show(): void {
        this.container.classList.add('visible');
    }

    public hide(): void {
        this.container.classList.remove('visible');
    }

    public toggle(): void {
        this.container.classList.toggle('visible');
    }

    public isReady(): boolean {
        return this.isInitialized;
    }

    public static getInstance(): ModularControlPanel {
        if (!ModularControlPanel.instance) {
            ModularControlPanel.instance = new ModularControlPanel(document.body);
        }
        return ModularControlPanel.instance;
    }

    public dispose(): void {
        this.unsubscribers.forEach(unsubscribe => unsubscribe());
        this.validationDisplay.dispose();
        if (this.updateTimeout !== null) {
            window.clearTimeout(this.updateTimeout);
        }
        this.container.remove();
        this.toggleButton.remove();
        ModularControlPanel.instance = null;
    }
}