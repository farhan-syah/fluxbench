//! Dependency Graph
//!
//! Manages dependencies between benchmarks, verifications, and synthetic metrics.

use fxhash::{FxHashMap, FxHashSet};
use thiserror::Error;

/// Errors from graph operations
#[derive(Debug, Error)]
pub enum GraphError {
    #[error("Cycle detected: {0}")]
    CycleDetected(String),

    #[error("Unknown node: {0}")]
    UnknownNode(String),
}

/// Dependency graph for execution ordering
#[derive(Debug, Default)]
pub struct DependencyGraph {
    /// Node -> dependencies mapping
    edges: FxHashMap<String, FxHashSet<String>>,
    /// All nodes
    nodes: FxHashSet<String>,
}

impl DependencyGraph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node
    pub fn add_node(&mut self, id: impl Into<String>) {
        self.nodes.insert(id.into());
    }

    /// Add a dependency: `from` depends on `to`
    pub fn add_dependency(&mut self, from: impl Into<String>, to: impl Into<String>) {
        let from = from.into();
        let to = to.into();

        self.nodes.insert(from.clone());
        self.nodes.insert(to.clone());

        self.edges.entry(from).or_default().insert(to);
    }

    /// Get dependencies for a node
    pub fn dependencies(&self, id: &str) -> Option<&FxHashSet<String>> {
        self.edges.get(id)
    }

    /// Perform topological sort
    ///
    /// Returns nodes in dependency order: dependencies come before dependents.
    pub fn topological_sort(&self) -> Result<Vec<String>, GraphError> {
        let mut result = Vec::new();
        let mut visited = FxHashSet::default();
        let mut temp_visited = FxHashSet::default();

        for node in &self.nodes {
            if !visited.contains(node) {
                self.visit(node, &mut visited, &mut temp_visited, &mut result)?;
            }
        }

        // Note: No reverse needed because our DFS visits dependencies first,
        // then adds the node. This naturally produces correct topological order.
        Ok(result)
    }

    fn visit(
        &self,
        node: &str,
        visited: &mut FxHashSet<String>,
        temp_visited: &mut FxHashSet<String>,
        result: &mut Vec<String>,
    ) -> Result<(), GraphError> {
        if temp_visited.contains(node) {
            return Err(GraphError::CycleDetected(node.to_string()));
        }

        if visited.contains(node) {
            return Ok(());
        }

        temp_visited.insert(node.to_string());

        if let Some(deps) = self.edges.get(node) {
            for dep in deps {
                self.visit(dep, visited, temp_visited, result)?;
            }
        }

        temp_visited.remove(node);
        visited.insert(node.to_string());
        result.push(node.to_string());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_sort() {
        let mut graph = DependencyGraph::new();

        graph.add_dependency("verify_net", "raw_bench");
        graph.add_dependency("verify_net", "overhead_bench");
        graph.add_node("raw_bench");
        graph.add_node("overhead_bench");

        let sorted = graph.topological_sort().unwrap();

        // raw_bench and overhead_bench should come before verify_net
        let raw_pos = sorted.iter().position(|x| x == "raw_bench").unwrap();
        let overhead_pos = sorted.iter().position(|x| x == "overhead_bench").unwrap();
        let verify_pos = sorted.iter().position(|x| x == "verify_net").unwrap();

        assert!(raw_pos < verify_pos);
        assert!(overhead_pos < verify_pos);
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = DependencyGraph::new();

        graph.add_dependency("a", "b");
        graph.add_dependency("b", "c");
        graph.add_dependency("c", "a"); // Creates cycle

        let result = graph.topological_sort();
        assert!(matches!(result, Err(GraphError::CycleDetected(_))));
    }
}
