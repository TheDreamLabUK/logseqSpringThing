<template>
    <div id="control-panel" :class="{ hidden: isHidden }">
        <button class="toggle-button" @click="togglePanel">
            {{ isHidden ? 'Show' : 'Hide' }} Controls
        </button>
        <div class="panel-content" v-show="!isHidden">
            <!-- Previous template content remains the same -->
        </div>
    </div>
</template>

<script>
import { defineComponent, ref, reactive, onMounted, onBeforeUnmount } from 'vue';
import { enableSpacemouse } from '../services/spacemouse.js';

export default defineComponent({
    name: 'ControlPanel',
    props: {
        websocketService: {
            type: Object,
            required: true
        }
    },
    setup(props, { emit }) {
        console.log('ControlPanel setup called');
        
        // Previous state declarations remain the same...

        // Methods
        const togglePanel = () => {
            isHidden.value = !isHidden.value;
            console.log(`ControlPanel is now ${isHidden.value ? 'hidden' : 'visible'}`);
        };

        const toggleGroup = (group) => {
            collapsedGroups[group] = !collapsedGroups[group];
            console.log(`Group "${group}" is now ${collapsedGroups[group] ? 'collapsed' : 'expanded'}`);
        };

        const emitChange = (name, value) => {
            console.log('Emitting control change:', name, value);
            emit('control-change', { name, value });
        };

        const handleSpacemouseToggle = () => {
            if (spacemouseEnabled.value) {
                enableSpacemouse();
                console.log('Spacemouse enabled');
            } else {
                console.log('Spacemouse disabled');
            }
            emitChange('spacemouseEnabled', spacemouseEnabled.value);
        };

        const applySettings = (settings) => {
            console.log('Applying settings:', settings);
            if (!settings) return;

            // Apply visualization settings
            if (settings.visualization) {
                const vis = settings.visualization;
                
                // Update node colors (convert from 0x to # format)
                if (vis.node_color) {
                    const nodeColorControl = nodeAppearanceControls.colors.find(c => c.name === 'nodeColor');
                    if (nodeColorControl) nodeColorControl.value = '#' + vis.node_color.substring(2);
                }
                if (vis.node_color_new) {
                    const control = nodeAppearanceControls.colors.find(c => c.name === 'nodeColorNew');
                    if (control) control.value = '#' + vis.node_color_new.substring(2);
                }
                if (vis.node_color_recent) {
                    const control = nodeAppearanceControls.colors.find(c => c.name === 'nodeColorRecent');
                    if (control) control.value = '#' + vis.node_color_recent.substring(2);
                }
                if (vis.node_color_medium) {
                    const control = nodeAppearanceControls.colors.find(c => c.name === 'nodeColorMedium');
                    if (control) control.value = '#' + vis.node_color_medium.substring(2);
                }
                if (vis.node_color_old) {
                    const control = nodeAppearanceControls.colors.find(c => c.name === 'nodeColorOld');
                    if (control) control.value = '#' + vis.node_color_old.substring(2);
                }
                if (vis.node_color_core) {
                    const control = nodeAppearanceControls.colors.find(c => c.name === 'nodeColorCore');
                    if (control) control.value = '#' + vis.node_color_core.substring(2);
                }
                if (vis.node_color_secondary) {
                    const control = nodeAppearanceControls.colors.find(c => c.name === 'nodeColorSecondary');
                    if (control) control.value = '#' + vis.node_color_secondary.substring(2);
                }

                // Update edge colors
                if (vis.edge_color) {
                    const control = edgeAppearanceControls.colors.find(c => c.name === 'edgeColor');
                    if (control) control.value = '#' + vis.edge_color.substring(2);
                }

                // Update hologram colors
                if (vis.hologram_color) {
                    const control = environmentControls.hologram.find(c => c.name === 'hologramColor');
                    if (control) control.value = '#' + vis.hologram_color.substring(2);
                }

                // Update node dimensions
                if (vis.min_node_size !== undefined) {
                    const control = nodeAppearanceControls.size.find(c => c.name === 'minNodeSize');
                    if (control) control.value = vis.min_node_size * 100; // Convert meters to cm
                }
                if (vis.max_node_size !== undefined) {
                    const control = nodeAppearanceControls.size.find(c => c.name === 'maxNodeSize');
                    if (control) control.value = vis.max_node_size * 100; // Convert meters to cm
                }

                // Update material properties
                if (vis.node_material_metalness !== undefined) {
                    const control = nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialMetalness');
                    if (control) control.value = vis.node_material_metalness;
                }
                if (vis.node_material_roughness !== undefined) {
                    const control = nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialRoughness');
                    if (control) control.value = vis.node_material_roughness;
                }
                if (vis.node_material_clearcoat !== undefined) {
                    const control = nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialClearcoat');
                    if (control) control.value = vis.node_material_clearcoat;
                }
                if (vis.node_material_clearcoat_roughness !== undefined) {
                    const control = nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialClearcoatRoughness');
                    if (control) control.value = vis.node_material_clearcoat_roughness;
                }
                if (vis.node_material_opacity !== undefined) {
                    const control = nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialOpacity');
                    if (control) control.value = vis.node_material_opacity;
                }

                // Update emissive properties
                if (vis.node_emissive_min_intensity !== undefined) {
                    const control = nodeAppearanceControls.emissive.find(c => c.name === 'nodeEmissiveMin');
                    if (control) control.value = vis.node_emissive_min_intensity;
                }
                if (vis.node_emissive_max_intensity !== undefined) {
                    const control = nodeAppearanceControls.emissive.find(c => c.name === 'nodeEmissiveMax');
                    if (control) control.value = vis.node_emissive_max_intensity;
                }

                // Update edge properties
                if (vis.edge_opacity !== undefined) {
                    const control = edgeAppearanceControls.properties.find(c => c.name === 'edgeOpacity');
                    if (control) control.value = vis.edge_opacity;
                }

                // Update hologram properties
                if (vis.hologram_scale !== undefined) {
                    const control = environmentControls.hologram.find(c => c.name === 'hologramScale');
                    if (control) control.value = vis.hologram_scale;
                }
                if (vis.hologram_opacity !== undefined) {
                    const control = environmentControls.hologram.find(c => c.name === 'hologramOpacity');
                    if (control) control.value = vis.hologram_opacity;
                }

                // Update fog properties
                if (vis.fog_density !== undefined) {
                    const control = environmentControls.fog.find(c => c.name === 'fogDensity');
                    if (control) control.value = vis.fog_density;
                }
            }

            // Apply bloom settings
            if (settings.bloom) {
                const bloom = settings.bloom;
                
                // Node bloom
                if (bloom.node_bloom_strength !== undefined) {
                    const control = bloomControls.node.find(c => c.name === 'nodeBloomStrength');
                    if (control) control.value = bloom.node_bloom_strength;
                }
                if (bloom.node_bloom_radius !== undefined) {
                    const control = bloomControls.node.find(c => c.name === 'nodeBloomRadius');
                    if (control) control.value = bloom.node_bloom_radius;
                }
                if (bloom.node_bloom_threshold !== undefined) {
                    const control = bloomControls.node.find(c => c.name === 'nodeBloomThreshold');
                    if (control) control.value = bloom.node_bloom_threshold;
                }

                // Edge bloom
                if (bloom.edge_bloom_strength !== undefined) {
                    const control = bloomControls.edge.find(c => c.name === 'edgeBloomStrength');
                    if (control) control.value = bloom.edge_bloom_strength;
                }
                if (bloom.edge_bloom_radius !== undefined) {
                    const control = bloomControls.edge.find(c => c.name === 'edgeBloomRadius');
                    if (control) control.value = bloom.edge_bloom_radius;
                }
                if (bloom.edge_bloom_threshold !== undefined) {
                    const control = bloomControls.edge.find(c => c.name === 'edgeBloomThreshold');
                    if (control) control.value = bloom.edge_bloom_threshold;
                }

                // Environment bloom
                if (bloom.environment_bloom_strength !== undefined) {
                    const control = bloomControls.environment.find(c => c.name === 'envBloomStrength');
                    if (control) control.value = bloom.environment_bloom_strength;
                }
                if (bloom.environment_bloom_radius !== undefined) {
                    const control = bloomControls.environment.find(c => c.name === 'envBloomRadius');
                    if (control) control.value = bloom.environment_bloom_radius;
                }
                if (bloom.environment_bloom_threshold !== undefined) {
                    const control = bloomControls.environment.find(c => c.name === 'envBloomThreshold');
                    if (control) control.value = bloom.environment_bloom_threshold;
                }
            }

            // Apply fisheye settings
            if (settings.fisheye) {
                const fisheye = settings.fisheye;
                
                fisheyeEnabled.value = fisheye.enabled || false;
                
                if (fisheye.strength !== undefined) {
                    const control = fisheyeControls.properties.find(c => c.name === 'fisheyeStrength');
                    if (control) control.value = fisheye.strength;
                }
                if (fisheye.radius !== undefined) {
                    const control = fisheyeControls.properties.find(c => c.name === 'fisheyeRadius');
                    if (control) control.value = fisheye.radius;
                }
                if (fisheye.focus_x !== undefined) {
                    const control = fisheyeControls.focus.find(c => c.name === 'fisheyeFocusX');
                    if (control) control.value = fisheye.focus_x;
                }
                if (fisheye.focus_y !== undefined) {
                    const control = fisheyeControls.focus.find(c => c.name === 'fisheyeFocusY');
                    if (control) control.value = fisheye.focus_y;
                }
                if (fisheye.focus_z !== undefined) {
                    const control = fisheyeControls.focus.find(c => c.name === 'fisheyeFocusZ');
                    if (control) control.value = fisheye.focus_z;
                }
            }

            // Apply physics settings
            if (settings.physics) {
                const physics = settings.physics;
                
                if (physics.simulation_mode) {
                    simulationMode.value = physics.simulation_mode;
                }

                // Update simulation parameters
                if (physics.force_directed_iterations !== undefined) {
                    const control = physicsControls.simulation.find(c => c.name === 'forceDirectedIterations');
                    if (control) control.value = physics.force_directed_iterations;
                }
                if (physics.force_directed_spring !== undefined) {
                    const control = physicsControls.simulation.find(c => c.name === 'forceDirectedSpring');
                    if (control) control.value = physics.force_directed_spring;
                }
                if (physics.force_directed_repulsion !== undefined) {
                    const control = physicsControls.simulation.find(c => c.name === 'forceDirectedRepulsion');
                    if (control) control.value = physics.force_directed_repulsion;
                }
                if (physics.force_directed_attraction !== undefined) {
                    const control = physicsControls.simulation.find(c => c.name === 'forceDirectedAttraction');
                    if (control) control.value = physics.force_directed_attraction;
                }
                if (physics.force_directed_damping !== undefined) {
                    const control = physicsControls.simulation.find(c => c.name === 'forceDirectedDamping');
                    if (control) control.value = physics.force_directed_damping;
                }

                // Update geometry parameters
                if (physics.geometry_min_segments !== undefined) {
                    const control = physicsControls.geometry.find(c => c.name === 'geometryMinSegments');
                    if (control) control.value = physics.geometry_min_segments;
                }
                if (physics.geometry_max_segments !== undefined) {
                    const control = physicsControls.geometry.find(c => c.name === 'geometryMaxSegments');
                    if (control) control.value = physics.geometry_max_segments;
                }
                if (physics.geometry_segment_per_hyperlink !== undefined) {
                    const control = physicsControls.geometry.find(c => c.name === 'geometrySegmentPerLink');
                    if (control) control.value = physics.geometry_segment_per_hyperlink;
                }
            }
        };

        const saveSettings = () => {
            // Collect all settings into TOML structure
            const settings = {
                visualization: {
                    // Node colors
                    node_color: nodeAppearanceControls.colors.find(c => c.name === 'nodeColor')?.value.replace('#', '0x'),
                    node_color_new: nodeAppearanceControls.colors.find(c => c.name === 'nodeColorNew')?.value.replace('#', '0x'),
                    node_color_recent: nodeAppearanceControls.colors.find(c => c.name === 'nodeColorRecent')?.value.replace('#', '0x'),
                    node_color_medium: nodeAppearanceControls.colors.find(c => c.name === 'nodeColorMedium')?.value.replace('#', '0x'),
                    node_color_old: nodeAppearanceControls.colors.find(c => c.name === 'nodeColorOld')?.value.replace('#', '0x'),
                    node_color_core: nodeAppearanceControls.colors.find(c => c.name === 'nodeColorCore')?.value.replace('#', '0x'),
                    node_color_secondary: nodeAppearanceControls.colors.find(c => c.name === 'nodeColorSecondary')?.value.replace('#', '0x'),

                    // Edge colors
                    edge_color: edgeAppearanceControls.colors.find(c => c.name === 'edgeColor')?.value.replace('#', '0x'),
                    
                    // Hologram
                    hologram_color: environmentControls.hologram.find(c => c.name === 'hologramColor')?.value.replace('#', '0x'),
                    hologram_scale: environmentControls.hologram.find(c => c.name === 'hologramScale')?.value,
                    hologram_opacity: environmentControls.hologram.find(c => c.name === 'hologramOpacity')?.value,

                    // Node dimensions
                    min_node_size: nodeAppearanceControls.size.find(c => c.name === 'minNodeSize')?.value / 100, // Convert cm to meters
                    max_node_size: nodeAppearanceControls.size.find(c => c.name === 'maxNodeSize')?.value / 100, // Convert cm to meters
                    
                    // Edge properties
                    edge_opacity: edgeAppearanceControls.properties.find(c => c.name === 'edgeOpacity')?.value,

                    // Material properties
                    node_material_metalness: nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialMetalness')?.value,
                    node_material_roughness: nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialRoughness')?.value,
                    node_material_clearcoat: nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialClearcoat')?.value,
                    node_material_clearcoat_roughness: nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialClearcoatRoughness')?.value,
                    node_material_opacity: nodeAppearanceControls.material.find(c => c.name === 'nodeMaterialOpacity')?.value,
                    
                    // Emissive properties
                    node_emissive_min_intensity: nodeAppearanceControls.emissive.find(c => c.name === 'nodeEmissiveMin')?.value,
                    node_emissive_max_intensity: nodeAppearanceControls.emissive.find(c => c.name === 'nodeEmissiveMax')?.value,

                    // Environment
                    fog_density: environmentControls.fog.find(c => c.name === 'fogDensity')?.value,
                },
                bloom: {
                    // Node bloom
                    node_bloom_strength: bloomControls.node.find(c => c.name === 'nodeBloomStrength')?.value,
                    node_bloom_radius: bloomControls.node.find(c => c.name === 'nodeBloomRadius')?.value,
                    node_bloom_threshold: bloomControls.node.find(c => c.name === 'nodeBloomThreshold')?.value,
                    
                    // Edge bloom
                    edge_bloom_strength: bloomControls.edge.find(c => c.name === 'edgeBloomStrength')?.value,
                    edge_bloom_radius: bloomControls.edge.find(c => c.name === 'edgeBloomRadius')?.value,
                    edge_bloom_threshold: bloomControls.edge.find(c => c.name === 'edgeBloomThreshold')?.value,
                    
                    // Environment bloom
                    environment_bloom_strength: bloomControls.environment.find(c => c.name === 'envBloomStrength')?.value,
                    environment_bloom_radius: bloomControls.environment.find(c => c.name === 'envBloomRadius')?.value,
                    environment_bloom_threshold: bloomControls.environment.find(c => c.name === 'envBloomThreshold')?.value
                },
                fisheye: {
                    enabled: fisheyeEnabled.value,
                    strength: fisheyeControls.properties.find(c => c.name === 'fisheyeStrength')?.value,
                    radius: fisheyeControls.properties.find(c => c.name === 'fisheyeRadius')?.value,
                    focus_x: fisheyeControls.focus.find(c => c.name === 'fisheyeFocusX')?.value,
                    focus_y: fisheyeControls.focus.find(c => c.name === 'fisheyeFocusY')?.value,
                    focus_z: fisheyeControls.focus.find(c => c.name === 'fisheyeFocusZ')?.value
                },
                physics: {
                    simulation_mode: simulationMode.value,
                    force_directed_iterations: physicsControls.simulation.find(c => c.name === 'forceDirectedIterations')?.value,
                    force_directed_spring: physicsControls.simulation.find(c => c.name === 'forceDirectedSpring')?.value,
                    force_directed_repulsion: physicsControls.simulation.find(c => c.name === 'forceDirectedRepulsion')?.value,
                    force_directed_attraction: physicsControls.simulation.find(c => c.name === 'forceDirectedAttraction')?.value,
                    force_directed_damping: physicsControls.simulation.find(c => c.name === 'forceDirectedDamping')?.value,
                    geometry_min_segments: physicsControls.geometry.find(c => c.name === 'geometryMinSegments')?.value,
                    geometry_max_segments: physicsControls.geometry.find(c => c.name === 'geometryMaxSegments')?.value,
                    geometry_segment_per_hyperlink: physicsControls.geometry.find(c => c.name === 'geometrySegmentPerLink')?.value
                }
            };

            console.log('Saving settings:', settings);
            props.websocketService.send({
                type: 'updateSettings',
                settings
            });
        };

        // Setup event listeners
        onMounted(() => {
            console.log('ControlPanel mounted');
            
            // Request initial settings
            props.websocketService.send({ type: 'requestSettings' });

            // Setup event listeners
            props.websocketService.on('serverSettings', (settings) => {
                console.log('Received server settings:', settings);
                applySettings(settings);
            });

            props.websocketService.on('settingsUpdated', (settings) => {
                console.log('Settings updated:', settings);
                applySettings(settings);
            });
        });

        onBeforeUnmount(() => {
            console.log('ControlPanel unmounting');
            props.websocketService.off('serverSettings');
            props.websocketService.off('settingsUpdated');
        });

        // Return all reactive state and methods
        return {
            isHidden,
            collapsedGroups,
            nodeAppearanceControls,
            edgeAppearanceControls,
            bloomControls,
            physicsControls,
            fisheyeEnabled,
            fisheyeControls,
            environmentControls,
            simulationMode,
            spacemouseEnabled,
            togglePanel,
            toggleGroup,
            emitChange,
            handleSpacemouseToggle,
            saveSettings
        };
    }
});
</script>

<style scoped>
/* Previous style section remains unchanged */
</style>
