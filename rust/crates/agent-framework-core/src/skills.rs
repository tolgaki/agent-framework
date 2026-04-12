// Copyright (c) Microsoft. All rights reserved.

//! Agent skills system for progressive disclosure of capabilities.
//!
//! Skills are named, self-describing capabilities that an agent can advertise
//! and execute on demand. They mirror .NET's `AgentSkill` /
//! `AgentSkillsProvider` and Python's `Skill` / `SkillsProvider`.
//!
//! # Architecture
//!
//! - A [`Skill`] has a name, description, optional instructions, and resources.
//! - A [`SkillsProvider`] aggregates skills from multiple [`SkillSource`]s
//!   and integrates with the agent as a [`ContextProvider`].
//! - Built-in sources: [`InMemorySkillSource`] (code-defined) and
//!   [`FileSkillSource`] (file-based discovery).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;

use crate::context::{AIContext, ContextProvider};
use crate::error::{AgentError, AgentResult};
use crate::session::AgentSession;

// ---------------------------------------------------------------------------
// Skill types
// ---------------------------------------------------------------------------

/// A named, self-describing agent capability.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Unique skill identifier.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// Detailed instructions injected into the system prompt when the skill
    /// is activated.
    pub instructions: Option<String>,
    /// Supplementary resources (code snippets, templates, data).
    pub resources: Vec<SkillResource>,
}

/// Supplementary content attached to a skill.
#[derive(Debug, Clone)]
pub struct SkillResource {
    /// Resource identifier (file name, key, etc.).
    pub name: String,
    /// MIME type hint.
    pub media_type: Option<String>,
    /// The resource content.
    pub content: String,
}

/// A script-based skill that can be executed.
#[derive(Debug, Clone)]
pub struct SkillScript {
    /// The skill this script belongs to.
    pub skill: Skill,
    /// The script content (e.g., a prompt template).
    pub script: String,
}

// ---------------------------------------------------------------------------
// Skill source trait
// ---------------------------------------------------------------------------

/// A source that provides skills.
///
/// Mirrors .NET's `AgentSkillsSource`.
#[async_trait]
pub trait SkillSource: Send + Sync {
    /// List all available skills from this source.
    async fn list_skills(&self) -> AgentResult<Vec<Skill>>;

    /// Get a specific skill by name, if it exists.
    async fn get_skill(&self, name: &str) -> AgentResult<Option<Skill>>;
}

// ---------------------------------------------------------------------------
// In-memory skill source
// ---------------------------------------------------------------------------

/// A skill source backed by an in-memory list.
pub struct InMemorySkillSource {
    skills: Vec<Skill>,
}

impl InMemorySkillSource {
    pub fn new() -> Self {
        Self { skills: Vec::new() }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, skill: Skill) -> Self {
        self.skills.push(skill);
        self
    }
}

impl Default for InMemorySkillSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SkillSource for InMemorySkillSource {
    async fn list_skills(&self) -> AgentResult<Vec<Skill>> {
        Ok(self.skills.clone())
    }

    async fn get_skill(&self, name: &str) -> AgentResult<Option<Skill>> {
        Ok(self.skills.iter().find(|s| s.name == name).cloned())
    }
}

// ---------------------------------------------------------------------------
// File-based skill source
// ---------------------------------------------------------------------------

/// Discovers skills from `SKILL.md` files in a directory tree.
///
/// Each skill file uses a frontmatter format:
/// ```markdown
/// ---
/// name: my-skill
/// description: Does something useful
/// ---
///
/// Detailed instructions for the model...
/// ```
///
/// Mirrors Python's file-based skill discovery.
pub struct FileSkillSource {
    root: PathBuf,
}

impl FileSkillSource {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn parse_skill_file(path: &Path) -> AgentResult<Skill> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| AgentError::InvalidRequest(format!("failed to read skill file {}: {e}", path.display())))?;

        // Prevent path traversal in resource references.
        let canonical = path
            .canonicalize()
            .map_err(|e| AgentError::InvalidRequest(format!("invalid skill path {}: {e}", path.display())))?;

        let (frontmatter, body) = parse_frontmatter(&content);

        let name = frontmatter
            .get("name")
            .cloned()
            .unwrap_or_else(|| {
                canonical
                    .parent()
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unnamed".to_string())
            });

        let description = frontmatter
            .get("description")
            .cloned()
            .unwrap_or_else(|| name.clone());

        Ok(Skill {
            name,
            description,
            instructions: if body.trim().is_empty() { None } else { Some(body) },
            resources: Vec::new(),
        })
    }
}

#[async_trait]
impl SkillSource for FileSkillSource {
    async fn list_skills(&self) -> AgentResult<Vec<Skill>> {
        let mut skills = Vec::new();
        if !self.root.is_dir() {
            return Ok(skills);
        }
        // Walk directory tree looking for SKILL.md files.
        fn walk(dir: &Path, skills: &mut Vec<Skill>) -> AgentResult<()> {
            let entries = std::fs::read_dir(dir)
                .map_err(|e| AgentError::InvalidRequest(format!("failed to read dir {}: {e}", dir.display())))?;
            for entry in entries {
                let entry = entry.map_err(|e| AgentError::InvalidRequest(e.to_string()))?;
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, skills)?;
                } else if path.file_name().map(|n| n == "SKILL.md").unwrap_or(false) {
                    skills.push(FileSkillSource::parse_skill_file(&path)?);
                }
            }
            Ok(())
        }
        walk(&self.root, &mut skills)?;
        Ok(skills)
    }

    async fn get_skill(&self, name: &str) -> AgentResult<Option<Skill>> {
        let skills = self.list_skills().await?;
        Ok(skills.into_iter().find(|s| s.name == name))
    }
}

/// Parse simple YAML-like frontmatter from a `---` delimited block.
fn parse_frontmatter(content: &str) -> (HashMap<String, String>, String) {
    let mut map = HashMap::new();
    let trimmed = content.trim();

    if !trimmed.starts_with("---") {
        return (map, content.to_string());
    }

    let after_first = &trimmed[3..];
    if let Some(end) = after_first.find("---") {
        let frontmatter = &after_first[..end];
        let body = &after_first[end + 3..];

        for line in frontmatter.lines() {
            let line = line.trim();
            if let Some((key, value)) = line.split_once(':') {
                map.insert(key.trim().to_string(), value.trim().to_string());
            }
        }

        return (map, body.to_string());
    }

    (map, content.to_string())
}

// ---------------------------------------------------------------------------
// Skills provider (context provider integration)
// ---------------------------------------------------------------------------

/// Aggregates skills from multiple sources and integrates with the agent
/// as a [`ContextProvider`].
///
/// When used as a context provider, it advertises available skills in the
/// system prompt so the model knows what capabilities are available.
pub struct SkillsProvider {
    sources: Vec<Box<dyn SkillSource>>,
    /// Optional list of skill names to activate (inject full instructions).
    /// If empty, only the skill directory is advertised.
    active_skills: Vec<String>,
}

impl SkillsProvider {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            active_skills: Vec::new(),
        }
    }

    pub fn add_source(mut self, source: impl SkillSource + 'static) -> Self {
        self.sources.push(Box::new(source));
        self
    }

    /// Activate specific skills by name (their full instructions will be
    /// injected into the system prompt).
    pub fn activate(mut self, skills: Vec<String>) -> Self {
        self.active_skills = skills;
        self
    }

    /// List all available skills across all sources.
    pub async fn all_skills(&self) -> AgentResult<Vec<Skill>> {
        let mut all = Vec::new();
        for source in &self.sources {
            all.extend(source.list_skills().await?);
        }
        Ok(all)
    }

    /// Find a skill by name across all sources.
    pub async fn find_skill(&self, name: &str) -> AgentResult<Option<Skill>> {
        for source in &self.sources {
            if let Some(skill) = source.get_skill(name).await? {
                return Ok(Some(skill));
            }
        }
        Ok(None)
    }
}

impl Default for SkillsProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ContextProvider for SkillsProvider {
    async fn provide_context(&self, _session: &AgentSession) -> AgentResult<AIContext> {
        let skills = self.all_skills().await?;
        if skills.is_empty() {
            return Ok(AIContext::default());
        }

        let mut instructions = String::from("## Available Skills\n\n");
        for skill in &skills {
            instructions.push_str(&format!("- **{}**: {}\n", skill.name, skill.description));
        }

        // Inject full instructions for active skills.
        for name in &self.active_skills {
            if let Some(skill) = skills.iter().find(|s| s.name == *name) {
                if let Some(inst) = &skill.instructions {
                    instructions.push_str(&format!("\n### Skill: {}\n\n{}\n", skill.name, inst));
                }
            }
        }

        Ok(AIContext {
            instructions: Some(instructions),
            messages: Vec::new(),
            tools: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_frontmatter_extracts_fields() {
        let content = "---\nname: test-skill\ndescription: A test\n---\nBody here";
        let (fm, body) = parse_frontmatter(content);
        assert_eq!(fm.get("name").unwrap(), "test-skill");
        assert_eq!(fm.get("description").unwrap(), "A test");
        assert!(body.contains("Body here"));
    }

    #[test]
    fn parse_frontmatter_no_delimiter() {
        let content = "Just plain text";
        let (fm, body) = parse_frontmatter(content);
        assert!(fm.is_empty());
        assert_eq!(body, content);
    }

    #[tokio::test]
    async fn in_memory_source_works() {
        let source = InMemorySkillSource::new()
            .add(Skill {
                name: "greet".to_string(),
                description: "Greet users".to_string(),
                instructions: Some("Always say hello first.".to_string()),
                resources: Vec::new(),
            });

        let skills = source.list_skills().await.unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "greet");

        let found = source.get_skill("greet").await.unwrap();
        assert!(found.is_some());

        let missing = source.get_skill("nonexistent").await.unwrap();
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn skills_provider_aggregates_sources() {
        let source1 = InMemorySkillSource::new().add(Skill {
            name: "a".to_string(),
            description: "Skill A".to_string(),
            instructions: None,
            resources: Vec::new(),
        });
        let source2 = InMemorySkillSource::new().add(Skill {
            name: "b".to_string(),
            description: "Skill B".to_string(),
            instructions: None,
            resources: Vec::new(),
        });

        let provider = SkillsProvider::new().add_source(source1).add_source(source2);
        let all = provider.all_skills().await.unwrap();
        assert_eq!(all.len(), 2);
    }
}
