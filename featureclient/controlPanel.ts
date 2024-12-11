import * as THREE from 'three'

// Types
interface ControlGroup {
    name: string
    label: string
    controls: Control[]
}

interface Control {
    name: string
    label: string
    type: 'color' | 'range' | 'checkbox'
    value: any
    min?: number
    max?: number
    step?: number
}

interface Settings {
    nodeSize: number
    nodeColor: string
    edgeWidth: number
    edgeColor: string
    opacity: number
    bloom: {
        enabled: boolean
        strength: number
        radius: number
        threshold: number
    }
    fisheye: {
        enabled: boolean
        strength: number
        radius: number
    }
}

// Default settings
const defaultSettings: Settings = {
    nodeSize: 2.5,
    nodeColor: '#4CAF50',
    edgeWidth: 0.25,
    edgeColor: '#E0E0E0',
    opacity: 0.7,
    bloom: {
        enabled: true,
        strength: 1.5,
        radius: 0.75,
        threshold: 0.3
    },
    fisheye: {
        enabled: false,
        strength: 2.0,
        radius: 200
    }
}

export class ControlPanel {
    private panel: HTMLElement
    private settings: Settings
    private onSettingsChange: (settings: Settings) => void
    private collapsedGroups: { [key: string]: boolean } = {}

    constructor(containerId: string, onSettingsChange: (settings: Settings) => void) {
        this.settings = { ...defaultSettings }
        this.onSettingsChange = onSettingsChange
        
        // Create panel
        this.panel = document.createElement('div')
        this.panel.id = 'control-panel'
        document.getElementById(containerId)?.appendChild(this.panel)
        
        this.initializePanel()
        this.loadSettings()
    }

    private initializePanel() {
        // Add toggle button
        const toggleButton = document.createElement('button')
        toggleButton.className = 'toggle-button'
        toggleButton.textContent = 'Hide Controls'
        toggleButton.onclick = () => this.togglePanel()
        this.panel.appendChild(toggleButton)

        // Add content container
        const content = document.createElement('div')
        content.className = 'panel-content'
        this.panel.appendChild(content)

        // Create control groups
        this.createControlGroups().forEach(group => {
            const groupElement = this.createGroupElement(group)
            content.appendChild(groupElement)
        })

        // Add styles
        this.addStyles()
    }

    private createControlGroups(): ControlGroup[] {
        return [
            {
                name: 'appearance',
                label: 'Node Appearance',
                controls: [
                    {
                        name: 'nodeSize',
                        label: 'Node Size',
                        type: 'range',
                        value: this.settings.nodeSize,
                        min: 0.5,
                        max: 5,
                        step: 0.1
                    },
                    {
                        name: 'nodeColor',
                        label: 'Node Color',
                        type: 'color',
                        value: this.settings.nodeColor
                    },
                    {
                        name: 'opacity',
                        label: 'Opacity',
                        type: 'range',
                        value: this.settings.opacity,
                        min: 0,
                        max: 1,
                        step: 0.1
                    }
                ]
            },
            {
                name: 'edges',
                label: 'Edge Appearance',
                controls: [
                    {
                        name: 'edgeWidth',
                        label: 'Edge Width',
                        type: 'range',
                        value: this.settings.edgeWidth,
                        min: 0.1,
                        max: 1,
                        step: 0.05
                    },
                    {
                        name: 'edgeColor',
                        label: 'Edge Color',
                        type: 'color',
                        value: this.settings.edgeColor
                    }
                ]
            },
            {
                name: 'bloom',
                label: 'Bloom Effect',
                controls: [
                    {
                        name: 'bloomEnabled',
                        label: 'Enable Bloom',
                        type: 'checkbox',
                        value: this.settings.bloom.enabled
                    },
                    {
                        name: 'bloomStrength',
                        label: 'Bloom Strength',
                        type: 'range',
                        value: this.settings.bloom.strength,
                        min: 0,
                        max: 3,
                        step: 0.1
                    },
                    {
                        name: 'bloomRadius',
                        label: 'Bloom Radius',
                        type: 'range',
                        value: this.settings.bloom.radius,
                        min: 0,
                        max: 1,
                        step: 0.05
                    }
                ]
            },
            {
                name: 'fisheye',
                label: 'Fisheye Effect',
                controls: [
                    {
                        name: 'fisheyeEnabled',
                        label: 'Enable Fisheye',
                        type: 'checkbox',
                        value: this.settings.fisheye.enabled
                    },
                    {
                        name: 'fisheyeStrength',
                        label: 'Fisheye Strength',
                        type: 'range',
                        value: this.settings.fisheye.strength,
                        min: 1,
                        max: 5,
                        step: 0.1
                    },
                    {
                        name: 'fisheyeRadius',
                        label: 'Fisheye Radius',
                        type: 'range',
                        value: this.settings.fisheye.radius,
                        min: 100,
                        max: 500,
                        step: 10
                    }
                ]
            }
        ]
    }

    private createGroupElement(group: ControlGroup): HTMLElement {
        const groupElement = document.createElement('div')
        groupElement.className = 'control-group'

        // Create header
        const header = document.createElement('div')
        header.className = 'group-header'
        header.onclick = () => this.toggleGroup(group.name)

        const title = document.createElement('h3')
        title.textContent = group.label
        header.appendChild(title)
        groupElement.appendChild(header)

        // Create content
        const content = document.createElement('div')
        content.className = 'group-content'
        if (this.collapsedGroups[group.name]) {
            content.style.display = 'none'
        }

        // Add controls
        group.controls.forEach(control => {
            const controlElement = this.createControlElement(control)
            content.appendChild(controlElement)
        })

        groupElement.appendChild(content)
        return groupElement
    }

    private createControlElement(control: Control): HTMLElement {
        const container = document.createElement('div')
        container.className = 'control-item'

        const label = document.createElement('label')
        label.textContent = control.label
        container.appendChild(label)

        let input: HTMLInputElement
        switch (control.type) {
            case 'color':
                input = document.createElement('input')
                input.type = 'color'
                input.value = control.value
                input.onchange = (e) => this.handleControlChange(control.name, (e.target as HTMLInputElement).value)
                break

            case 'range':
                input = document.createElement('input')
                input.type = 'range'
                input.min = control.min?.toString() || '0'
                input.max = control.max?.toString() || '1'
                input.step = control.step?.toString() || '0.1'
                input.value = control.value
                input.oninput = (e) => this.handleControlChange(control.name, parseFloat((e.target as HTMLInputElement).value))

                const value = document.createElement('span')
                value.className = 'range-value'
                value.textContent = control.value.toFixed(2)
                container.appendChild(value)
                break

            case 'checkbox':
                input = document.createElement('input')
                input.type = 'checkbox'
                input.checked = control.value
                input.onchange = (e) => this.handleControlChange(control.name, (e.target as HTMLInputElement).checked)
                break

            default:
                throw new Error(`Unsupported control type: ${control.type}`)
        }

        container.appendChild(input)
        return container
    }

    private handleControlChange(name: string, value: any) {
        switch (name) {
            case 'nodeSize':
                this.settings.nodeSize = value
                break
            case 'nodeColor':
                this.settings.nodeColor = value
                break
            case 'edgeWidth':
                this.settings.edgeWidth = value
                break
            case 'edgeColor':
                this.settings.edgeColor = value
                break
            case 'opacity':
                this.settings.opacity = value
                break
            case 'bloomEnabled':
                this.settings.bloom.enabled = value
                break
            case 'bloomStrength':
                this.settings.bloom.strength = value
                break
            case 'bloomRadius':
                this.settings.bloom.radius = value
                break
            case 'fisheyeEnabled':
                this.settings.fisheye.enabled = value
                break
            case 'fisheyeStrength':
                this.settings.fisheye.strength = value
                break
            case 'fisheyeRadius':
                this.settings.fisheye.radius = value
                break
        }

        this.saveSettings()
        this.onSettingsChange(this.settings)
    }

    private togglePanel() {
        this.panel.classList.toggle('hidden')
        const button = this.panel.querySelector('.toggle-button') as HTMLButtonElement
        button.textContent = this.panel.classList.contains('hidden') ? 'Show Controls' : 'Hide Controls'
    }

    private toggleGroup(groupName: string) {
        this.collapsedGroups[groupName] = !this.collapsedGroups[groupName]
        const content = this.panel.querySelector(`[data-group="${groupName}"] .group-content`) as HTMLElement
        if (content) {
            content.style.display = this.collapsedGroups[groupName] ? 'none' : 'block'
        }
    }

    private saveSettings() {
        localStorage.setItem('graphSettings', JSON.stringify(this.settings))
    }

    private loadSettings() {
        const saved = localStorage.getItem('graphSettings')
        if (saved) {
            this.settings = { ...this.settings, ...JSON.parse(saved) }
            this.onSettingsChange(this.settings)
        }
    }

    private addStyles() {
        const style = document.createElement('style')
        style.textContent = `
            #control-panel {
                position: fixed;
                top: 20px;
                right: 0;
                width: 300px;
                max-height: 90vh;
                background-color: rgba(20, 20, 20, 0.9);
                color: #ffffff;
                border-radius: 10px 0 0 10px;
                overflow-y: auto;
                z-index: 1000;
                transition: transform 0.3s ease-in-out;
                box-shadow: -2px 0 10px rgba(0, 0, 0, 0.5);
            }

            #control-panel.hidden {
                transform: translateX(calc(100% - 40px));
            }

            .toggle-button {
                position: absolute;
                left: 0;
                top: 50%;
                transform: translateY(-50%) rotate(-90deg);
                transform-origin: left center;
                background-color: rgba(20, 20, 20, 0.9);
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                cursor: pointer;
                border-radius: 5px 5px 0 0;
                font-size: 0.9em;
                white-space: nowrap;
                z-index: 1001;
            }

            .panel-content {
                padding: 20px;
            }

            .control-group {
                margin-bottom: 16px;
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 6px;
                overflow: hidden;
            }

            .group-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px;
                background-color: rgba(255, 255, 255, 0.1);
                cursor: pointer;
            }

            .group-header h3 {
                margin: 0;
                font-size: 1em;
                font-weight: 500;
            }

            .group-content {
                padding: 12px;
            }

            .control-item {
                margin-bottom: 12px;
            }

            .control-item label {
                display: block;
                margin-bottom: 4px;
                font-size: 0.9em;
                color: #cccccc;
            }

            .control-item input[type="range"] {
                width: 100%;
                height: 6px;
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
                -webkit-appearance: none;
            }

            .control-item input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none;
                width: 16px;
                height: 16px;
                background-color: #ffffff;
                border-radius: 50%;
                cursor: pointer;
            }

            .control-item input[type="color"] {
                width: 100%;
                height: 30px;
                border: none;
                border-radius: 4px;
                background-color: transparent;
            }

            .range-value {
                float: right;
                font-size: 0.8em;
                color: #999999;
            }
        `
        document.head.appendChild(style)
    }
}
