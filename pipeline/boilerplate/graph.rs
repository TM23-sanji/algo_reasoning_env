// Graph helpers
// Provides adjacency list construction functions for graph problems

use std::collections::{HashMap, HashSet};

/// Build an undirected adjacency list from an edge list
/// 
/// Args:
///   n: Number of vertices (0 to n-1)
///   edges: List of edges [u, v], each edge added in both directions
/// 
/// Returns:
///   Vec<Vec<usize>> where adj[u] contains all neighbors of u
/// 
/// Example:
///   build_adj_list(4, vec![vec![0,1], vec![1,2], vec![2,3]])
///   Returns: [[1], [0,2], [1,3], [2]]
pub fn build_adj_list(n: usize, edges: Vec<Vec<i32>>) -> Vec<Vec<usize>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    
    for edge in edges {
        if edge.len() >= 2 {
            let u = edge[0] as usize;
            let v = edge[1] as usize;
            
            if u < n && v < n {
                adj[u].push(v);
                adj[v].push(u);
            }
        }
    }
    
    adj
}

/// Build a directed adjacency list from an edge list
/// 
/// Args:
///   n: Number of vertices (0 to n-1)
///   edges: List of directed edges [u, v], only u -> v
/// 
/// Returns:
///   Vec<Vec<usize>> where adj[u] contains all outgoing neighbors of u
/// 
/// Example:
///   build_directed_adj_list(4, vec![vec![0,1], vec![1,2], vec![2,3]])
///   Returns: [[1], [2], [3], []]
pub fn build_directed_adj_list(n: usize, edges: Vec<Vec<i32>>) -> Vec<Vec<usize>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    
    for edge in edges {
        if edge.len() >= 2 {
            let u = edge[0] as usize;
            let v = edge[1] as usize;
            
            if u < n && v < n {
                adj[u].push(v);
            }
        }
    }
    
    adj
}

/// Build a weighted undirected adjacency list
/// Returns Vec<Vec<(usize, i32)>> where each element is (neighbor, weight)
pub fn build_weighted_adj_list(n: usize, edges: Vec<Vec<i32>>) -> Vec<Vec<(usize, i32)>> {
    let mut adj: Vec<Vec<(usize, i32)>> = vec![Vec::new(); n];
    
    for edge in edges {
        if edge.len() >= 3 {
            let u = edge[0] as usize;
            let v = edge[1] as usize;
            let w = edge[2];
            
            if u < n && v < n {
                adj[u].push((v, w));
                adj[v].push((u, w));
            }
        }
    }
    
    adj
}

/// Build a weighted directed adjacency list
pub fn build_weighted_directed_adj_list(n: usize, edges: Vec<Vec<i32>>) -> Vec<Vec<(usize, i32)>> {
    let mut adj: Vec<Vec<(usize, i32)>> = vec![Vec::new(); n];
    
    for edge in edges {
        if edge.len() >= 3 {
            let u = edge[0] as usize;
            let v = edge[1] as usize;
            let w = edge[2];
            
            if u < n && v < n {
                adj[u].push((v, w));
            }
        }
    }
    
    adj
}

/// Detect a cycle in an undirected graph using DFS
pub fn has_cycle_undirected(n: usize, edges: Vec<Vec<i32>>) -> bool {
    let adj = build_adj_list(n, edges);
    let mut visited = vec![false; n];
    
    fn dfs(node: usize, parent: usize, adj: &[Vec<usize>], visited: &mut [bool]) -> bool {
        visited[node] = true;
        
        for &neighbor in &adj[node] {
            if !visited[neighbor] {
                if dfs(neighbor, node, adj, visited) {
                    return true;
                }
            } else if neighbor != parent {
                return true;
            }
        }
        
        false
    }
    
    for i in 0..n {
        if !visited[i] {
            if dfs(i, usize::MAX, &adj, &mut visited) {
                return true;
            }
        }
    }
    
    false
}

/// Detect a cycle in a directed graph using DFS (color-based)
pub fn has_cycle_directed(n: usize, edges: Vec<Vec<i32>>) -> bool {
    let adj = build_directed_adj_list(n, edges);
    // 0: white (unvisited), 1: gray (visiting), 2: black (visited)
    let mut color = vec![0; n];
    
    fn dfs(node: usize, adj: &[Vec<usize>], color: &mut [i32]) -> bool {
        color[node] = 1; // Mark as visiting (gray)
        
        for &neighbor in &adj[node] {
            if color[neighbor] == 1 {
                return true; // Back edge found (cycle)
            }
            if color[neighbor] == 0 && dfs(neighbor, adj, color) {
                return true;
            }
        }
        
        color[node] = 2; // Mark as visited (black)
        false
    }
    
    for i in 0..n {
        if color[i] == 0 {
            if dfs(i, &adj, &mut color) {
                return true;
            }
        }
    }
    
    false
}
