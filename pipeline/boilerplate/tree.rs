// Tree helpers
// Provides TreeNode struct and conversion functions for tree problems

use std::rc::Rc;
use std::cell::RefCell;
use std::collections::VecDeque;

#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

/// Convert a Vec<Option<i32>> into a TreeNode using level-order traversal
/// Format: [Some(1), None, Some(2), Some(3)] represents:
///     1
///    / \
///   -   2
///      / \
///     3   -
/// Trailing None values are automatically handled
/// 
/// Example: tree_from_vec(vec![Some(1), Some(2), Some(3)])
pub fn tree_from_vec(vals: Vec<Option<i32>>) -> Option<Rc<RefCell<TreeNode>>> {
    if vals.is_empty() || vals[0].is_none() {
        return None;
    }
    
    let root = Rc::new(RefCell::new(TreeNode::new(vals[0].unwrap())));
    let mut queue = VecDeque::new();
    queue.push_back(Rc::clone(&root));
    
    let mut i = 1;
    while !queue.is_empty() && i < vals.len() {
        let node = queue.pop_front().unwrap();
        let mut node_mut = node.borrow_mut();
        
        // Add left child
        if i < vals.len() {
            if let Some(val) = vals[i] {
                let left = Rc::new(RefCell::new(TreeNode::new(val)));
                node_mut.left = Some(Rc::clone(&left));
                queue.push_back(left);
            }
            i += 1;
        }
        
        // Add right child
        if i < vals.len() {
            if let Some(val) = vals[i] {
                let right = Rc::new(RefCell::new(TreeNode::new(val)));
                node_mut.right = Some(Rc::clone(&right));
                queue.push_back(right);
            }
            i += 1;
        }
    }
    
    Some(root)
}

/// Convert a TreeNode into Vec<Option<i32>> using level-order traversal
/// Returns values in level-order with None for missing nodes
/// Trailing None values are stripped for LeetCode compatibility
/// 
/// Example:     1         →  vec![Some(1), Some(2), Some(3)]
///            /   \
///           2     3
pub fn tree_to_vec(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<Option<i32>> {
    let mut result = Vec::new();
    
    if let Some(root) = root {
        let mut queue = VecDeque::new();
        queue.push_back(Some(Rc::clone(&root)));
        
        while !queue.is_empty() {
            let node = queue.pop_front().unwrap();
            
            match node {
                Some(n) => {
                    let n_borrow = n.borrow();
                    result.push(Some(n_borrow.val));
                    queue.push_back(n_borrow.left.as_ref().map(Rc::clone));
                    queue.push_back(n_borrow.right.as_ref().map(Rc::clone));
                }
                None => {
                    result.push(None);
                }
            }
        }
    }
    
    // Strip trailing None values for LeetCode format
    while result.last() == Some(&None) {
        result.pop();
    }
    
    result
}

/// In-order traversal of tree (left, root, right)
pub fn inorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut result = Vec::new();
    
    fn traverse(node: Option<Rc<RefCell<TreeNode>>>, result: &mut Vec<i32>) {
        if let Some(n) = node {
            let n_borrow = n.borrow();
            traverse(n_borrow.left.as_ref().map(Rc::clone), result);
            result.push(n_borrow.val);
            traverse(n_borrow.right.as_ref().map(Rc::clone), result);
        }
    }
    
    traverse(root, &mut result);
    result
}

/// Pre-order traversal of tree (root, left, right)
pub fn preorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut result = Vec::new();
    
    fn traverse(node: Option<Rc<RefCell<TreeNode>>>, result: &mut Vec<i32>) {
        if let Some(n) = node {
            let n_borrow = n.borrow();
            result.push(n_borrow.val);
            traverse(n_borrow.left.as_ref().map(Rc::clone), result);
            traverse(n_borrow.right.as_ref().map(Rc::clone), result);
        }
    }
    
    traverse(root, &mut result);
    result
}

/// Post-order traversal of tree (left, right, root)
pub fn postorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
    let mut result = Vec::new();
    
    fn traverse(node: Option<Rc<RefCell<TreeNode>>>, result: &mut Vec<i32>) {
        if let Some(n) = node {
            let n_borrow = n.borrow();
            traverse(n_borrow.left.as_ref().map(Rc::clone), result);
            traverse(n_borrow.right.as_ref().map(Rc::clone), result);
            result.push(n_borrow.val);
        }
    }
    
    traverse(root, &mut result);
    result
}
