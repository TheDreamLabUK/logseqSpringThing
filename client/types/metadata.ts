export interface NodeMetadata {
    name: string;
    commitAge: number;        // Age in days
    hyperlinkCount: number;   // Number of hyperlinks
    importance: number;       // Normalized importance (0-1)
    position: {
        x: number;
        y: number;
        z: number;
    };
}
