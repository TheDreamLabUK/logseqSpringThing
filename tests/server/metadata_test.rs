use chrono::prelude::*;
use logseq_spring_thing::models::metadata::Metadata;

#[test]
fn test_metadata_creation() {
    let now = Utc::now();
    let metadata = Metadata {
        file_name: String::from("test.md"),
        last_modified: now,
        processed_file: String::from("/path/to/processed/test.md"),
        original_file: String::from("/path/to/original/test.md"),
    };

    assert_eq!(metadata.file_name, "test.md");
    assert_eq!(metadata.last_modified, now);
    assert_eq!(metadata.processed_file, "/path/to/processed/test.md");
    assert_eq!(metadata.original_file, "/path/to/original/test.md");
}

#[test]
fn test_metadata_serialization() {
    let now = Utc::now();
    let metadata = Metadata {
        file_name: String::from("test.md"),
        last_modified: now,
        processed_file: String::from("/path/to/processed/test.md"),
        original_file: String::from("/path/to/original/test.md"),
    };

    let serialized = serde_json::to_string(&metadata).unwrap();
    let deserialized: Metadata = serde_json::from_str(&serialized).unwrap();

    assert_eq!(metadata.file_name, deserialized.file_name);
    assert_eq!(metadata.last_modified, deserialized.last_modified);
    assert_eq!(metadata.processed_file, deserialized.processed_file);
    assert_eq!(metadata.original_file, deserialized.original_file);
}

#[test]
fn test_metadata_comparison() {
    let now = Utc::now();
    let metadata1 = Metadata {
        file_name: String::from("test1.md"),
        last_modified: now,
        processed_file: String::from("/path/to/processed/test1.md"),
        original_file: String::from("/path/to/original/test1.md"),
    };

    let metadata2 = Metadata {
        file_name: String::from("test2.md"),
        last_modified: now,
        processed_file: String::from("/path/to/processed/test2.md"),
        original_file: String::from("/path/to/original/test2.md"),
    };

    assert_ne!(metadata1.file_name, metadata2.file_name);
    assert_eq!(metadata1.last_modified, metadata2.last_modified);
    assert_ne!(metadata1.processed_file, metadata2.processed_file);
    assert_ne!(metadata1.original_file, metadata2.original_file);
}

#[test]
fn test_metadata_default() {
    let default_metadata: Metadata = Default::default();

    assert_eq!(default_metadata.file_name, "");
    assert_eq!(default_metadata.processed_file, "");
    assert_eq!(default_metadata.original_file, "");
    // Note: We can't easily test the default last_modified as it's not a constant value
}