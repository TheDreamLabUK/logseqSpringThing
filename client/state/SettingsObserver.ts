import { Settings } from '../core/types';
import { createLogger } from '../core/logger';

const logger = createLogger('SettingsObserver');

export type SettingsChangeCallback = (settings: Settings) => void;

export class SettingsObserver {
    private static instance: SettingsObserver;
    private observers = new Map<string, SettingsChangeCallback>();

    private constructor() {}

    static getInstance(): SettingsObserver {
        if (!SettingsObserver.instance) {
            SettingsObserver.instance = new SettingsObserver();
        }
        return SettingsObserver.instance;
    }

    subscribe(id: string, callback: SettingsChangeCallback): () => void {
        logger.debug(`Subscribing observer: ${id}`);
        this.observers.set(id, callback);
        return () => this.unsubscribe(id);
    }

    unsubscribe(id: string): void {
        logger.debug(`Unsubscribing observer: ${id}`);
        this.observers.delete(id);
    }

    notifyAll(settings: Settings): void {
        logger.debug(`Notifying ${this.observers.size} observers of settings change`);
        this.observers.forEach((callback, id) => {
            try {
                callback(settings);
            } catch (error) {
                logger.error(`Error in settings observer ${id}:`, error);
            }
        });
    }
}
